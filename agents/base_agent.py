from __future__ import annotations

"""
Base agent class implementing the agentic tool-use loop via Gemini.

All specialized agents inherit from BaseAgent and register their tools
by overriding `_get_tools()` and `_execute_tool()`.

Tool definitions use OpenAI/Gemini format:
    {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from agents.perf import StageTimer
from agents.tool_gap_analyzer import analyze_for_tool_gap
from config.settings import get_settings
from models.conversation import AgentResponse

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10
# After exhausting all models in the pool, wait this many seconds before trying again.
_BACKOFF_DELAYS = (2, 4, 8)

# Exceptions that are worth rotating/retrying: transient infrastructure errors
# (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)
# plus NotFoundError — a model can be pulled from the API early/without notice
# (it happened to gemini-2.5-flash and gemini-2.5-flash-lite in 2026-07, well
# ahead of their announced shutdown date), and a dead model in the pool must
# not crash the whole call — just skip it and try the next one.
_RETRYABLE = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError, NotFoundError)


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop/repair message turns Gemini rejects outright.

    Gemini answers ANY request containing a message with an empty text part with
    a non-retryable 400 INVALID_ARGUMENT ("Unable to submit request because it
    has an empty text parameter"). A blank assistant turn gets into the stored
    conversation history whenever a model reply comes back empty (output
    truncated by max_tokens, a safety filter, or a tool round that produced no
    text) — and from then on EVERY later message in that thread replays that
    blank turn and fails the same way, so the whole conversation is dead until
    the user starts a new one. The orchestrator now refuses to persist a blank
    reply in the first place; this is the safety net that also heals threads
    already poisoned by one, since sessions live in Redis for 30 days.

    Rules:
      - user/assistant text turns with blank content → dropped entirely.
      - assistant turns carrying tool_calls → kept (the payload is in
        tool_calls), but a blank content string is normalized to None, which
        trips the same 400.
      - tool results → never dropped (an orphaned tool_call is itself a 400),
        blank content is replaced with a placeholder instead.
    """
    clean: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        blank = not content.strip() if isinstance(content, str) else content is None

        if msg.get("role") == "tool":
            clean.append({**msg, "content": "(no result)"} if blank else msg)
        elif msg.get("tool_calls"):
            clean.append({**msg, "content": None} if blank else msg)
        elif not blank:
            clean.append(msg)
        else:
            logger.warning("Dropping blank %s message before LLM call", msg.get("role"))
    return clean



# ── 400 fallback ladder ───────────────────────────────────────────────────────
# A 400 INVALID_ARGUMENT from Gemini is NOT retryable: rotating models or backing
# off just reproduces it, and the whole reply dies with "nie udało się przetworzyć
# tej wiadomości". Gemini's message for these is often the bare "Request contains
# an invalid argument.", naming no field — so the only way to find out which part
# of the payload it dislikes is to send it again without that part.
#
# Each quirk below is an optional piece of the request that some model/backend
# version rejects outright. On a 400 they are dropped one at a time, cumulatively,
# and the first variant that succeeds is remembered FOR THAT MODEL for the rest of
# the process — so the retry cost is paid once, not per request. Whichever quirk
# turns out to be needed is logged at ERROR level, which is what tells us (from
# production logs) what was actually wrong.
#
# Ordered most-likely-first:
#   reasoning_effort   — "none" asks the backend for a thinking budget of 0, which
#                        models that cannot disable thinking reject. Every LLM call
#                        in this app sends it (added in 03a9f9f to cut latency).
#   empty_tool_params  — a function declaration with "parameters": {"type":
#                        "object", "properties": {}}; Gemini requires an OBJECT
#                        schema to have properties, so those must be sent with no
#                        parameters key at all. 12 of the Allegro tools are like this.
#   tool_turns         — assistant turns carrying tool_calls plus their tool results.
#                        The interpret call replays them with NO tools declared, and
#                        thinking models additionally expect their own signature
#                        fields echoed back with a function call. Flattened to plain
#                        text, the model still sees every tool result.
#   leading_assistant  — a history that opens with an assistant turn instead of a
#                        user one. Possible since the invoice reminder writes itself
#                        into the conversation unprompted (main.py._record_assistant_turns),
#                        and some backends insist the first turn be the user's.
_QUIRK_ORDER = ("reasoning_effort", "empty_tool_params", "tool_turns", "leading_assistant")

# model → quirks already known to be rejected by it. Process-lifetime memo.
_MODEL_QUIRKS: dict[str, set[str]] = {}


def _strip_empty_tool_params(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tool in tools:
        fn = tool.get("function") or {}
        params = fn.get("parameters") or {}
        if params.get("type") == "object" and not params.get("properties"):
            fn = {k: v for k, v in fn.items() if k != "parameters"}
            tool = {**tool, "function": fn}
        out.append(tool)
    return out


def _flatten_tool_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite tool-call/tool-result turns as ordinary text turns, keeping the data."""
    out = []
    for msg in messages:
        if msg.get("role") == "tool":
            out.append({"role": "user", "content": f"[WYNIK NARZĘDZIA]\n{msg.get('content') or ''}"})
        elif msg.get("tool_calls"):
            names = ", ".join(
                (tc.get("function") or {}).get("name", "?") for tc in msg["tool_calls"]
            )
            out.append({
                "role": "assistant",
                "content": msg.get("content") or f"[Wywołano narzędzia: {names}]",
            })
        else:
            out.append(msg)
    return out


def _drop_leading_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the turns that precede the conversation's first user turn, keeping
    the system prompt. Loses the opening context, which is the point: it is only
    reached when the backend refused the request over it."""
    out = []
    seen_user = False
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            out.append(msg)
            continue
        if not seen_user:
            if role != "user":
                continue
            seen_user = True
        out.append(msg)
    return out


def _apply_quirk(api_kwargs: dict[str, Any], quirk: str) -> dict[str, Any]:
    kwargs = dict(api_kwargs)
    if quirk == "reasoning_effort":
        kwargs.pop("reasoning_effort", None)
    elif quirk == "empty_tool_params" and kwargs.get("tools"):
        kwargs["tools"] = _strip_empty_tool_params(kwargs["tools"])
    elif quirk == "tool_turns" and kwargs.get("messages"):
        kwargs["messages"] = _flatten_tool_turns(kwargs["messages"])
    elif quirk == "leading_assistant" and kwargs.get("messages"):
        kwargs["messages"] = _drop_leading_assistant(kwargs["messages"])
    return kwargs


def _payload_shape(model: str, api_kwargs: dict[str, Any]) -> str:
    """Compact description of a request, for the log line left behind when even the
    last fallback 400s — enough to tell which call it was and what it carried."""
    messages = api_kwargs.get("messages") or []
    roles = ",".join(
        f"{m.get('role')}{'+tc' if m.get('tool_calls') else ''}" for m in messages
    )
    return (
        f"model={model} tools={len(api_kwargs.get('tools') or [])} "
        f"tool_choice={api_kwargs.get('tool_choice')} "
        f"reasoning_effort={api_kwargs.get('reasoning_effort')} "
        f"max_tokens={api_kwargs.get('max_tokens')} msgs=[{roles}]"
    )


async def _create_with_fallbacks(
    client: AsyncOpenAI,
    model: str,
    label: str,
    api_kwargs: dict[str, Any],
):
    """One model attempt, walking the quirk ladder above on a 400."""
    known = _MODEL_QUIRKS.setdefault(model, set())
    kwargs = api_kwargs
    for quirk in _QUIRK_ORDER:
        if quirk in known:
            kwargs = _apply_quirk(kwargs, quirk)

    try:
        return await client.chat.completions.create(model=model, **kwargs)
    except _RETRYABLE:
        raise
    except APIStatusError as exc:
        if exc.status_code != 400:
            raise
        last_exc: APIStatusError = exc

    for quirk in _QUIRK_ORDER:
        if quirk in known:
            continue
        kwargs = _apply_quirk(kwargs, quirk)
        try:
            response = await client.chat.completions.create(model=model, **kwargs)
        except _RETRYABLE:
            raise
        except APIStatusError as exc:
            if exc.status_code != 400:
                raise
            last_exc = exc
            continue
        known.add(quirk)
        logger.error(
            "%s: %s rejected the request with 400; it succeeded after dropping %r — "
            "dropping it for this model from now on. Consider removing it for good.",
            label, model, quirk,
        )
        return response

    logger.error(
        "%s: %s returned 400 for every fallback variant | %s | error=%s",
        label, model, _payload_shape(model, api_kwargs), last_exc,
    )
    raise last_exc


async def _call_with_retry(
    client: AsyncOpenAI,
    model_pool: list[str],
    label: str,
    **api_kwargs,
):
    """
    Call client.chat.completions.create with model rotation + exponential backoff.

    Strategy:
      1. Try each model in model_pool in order.
         On a retryable error (429, 500, connection/timeout) → rotate to next model.
      2. After all models exhausted → wait (exponential backoff) → restart from first model.
      3. After all backoff rounds exhausted → raise the last seen exception.

    A 400 is never retryable, so it is handled separately by the quirk ladder in
    _create_with_fallbacks rather than by rotation.
    """
    if "messages" in api_kwargs:
        api_kwargs["messages"] = sanitize_messages(api_kwargs["messages"])

    last_exc: Exception = RuntimeError("No models in pool")
    for backoff_round, delay in enumerate((*_BACKOFF_DELAYS, None)):
        for model in model_pool:
            try:
                logger.debug("%s: calling model %s", label, model)
                return await _create_with_fallbacks(client, model, label, api_kwargs)
            except _RETRYABLE as exc:
                last_exc = exc
                logger.warning(
                    "%s: %s on %s, rotating to next model",
                    label, type(exc).__name__, model,
                )

        # All models in pool returned retryable errors
        if delay is None:
            logger.error(
                "%s: all models failed after %d rounds, giving up",
                label, backoff_round + 1,
            )
            raise last_exc
        logger.warning(
            "%s: full pool failed (round %d), backing off %ds",
            label, backoff_round + 1, delay,
        )
        await asyncio.sleep(delay)


# ── Blank replies ─────────────────────────────────────────────────────────────
# A 200 OK whose message carries neither text nor a tool call. Gemini returns
# one when the visible answer never got emitted: the budget went on invisible
# reasoning and finish_reason came back "length" (thinking-capable models —
# gemini-3.5-flash, 2.5-flash — do this whenever reasoning_effort="none" was
# rejected and dropped by the quirk ladder above), a safety filter dropped the
# candidate, or the model simply had nothing to say about the tool results it
# was handed. It is not an exception, so none of the rotation/backoff above ever
# fires — the empty string travels all the way to the user as "Przepraszam, nie
# udało się wygenerować odpowiedzi…" (agents/orchestrator.py), with a single
# WARNING that says nothing about the cause. Hence both halves below: log what
# the API actually reported, and give another model in the pool a turn.


def is_blank_reply(response: Any) -> bool:
    """True when the model returned neither text nor a tool call."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return True
    msg = getattr(choices[0], "message", None)
    if getattr(msg, "tool_calls", None):
        return False
    content = getattr(msg, "content", None)
    return not (content or "").strip()


def _blank_reply_shape(response: Any) -> str:
    """What the API said it did with the token budget — the why-was-it-blank log line."""
    choice = (getattr(response, "choices", None) or [None])[0]
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)

    def _num(obj: Any, attr: str) -> Any:
        value = getattr(obj, attr, None)
        return value if isinstance(value, int) else "?"

    model = getattr(response, "model", None)
    finish = getattr(choice, "finish_reason", None)
    return (
        f"model={model if isinstance(model, str) else '?'} "
        f"finish_reason={finish if isinstance(finish, str) else '?'} "
        f"prompt_tokens={_num(usage, 'prompt_tokens')} "
        f"completion_tokens={_num(usage, 'completion_tokens')} "
        f"reasoning_tokens={_num(details, 'reasoning_tokens')}"
    )


async def _call_for_reply(
    client: AsyncOpenAI,
    model_pool: list[str],
    label: str,
    **api_kwargs,
):
    """`_call_with_retry` for calls that must come back with something.

    Same rotation/backoff on errors, plus one extra attempt on a *different*
    model when the reply is blank (see above) — a blank reply is a property of
    the model that produced it, so re-sending to the same one is pointless, and
    the pool already exists precisely so one model's bad day is not the answer's.
    """
    pool = list(model_pool)
    response = await _call_with_retry(client, pool, label, **api_kwargs)
    if not is_blank_reply(response):
        return response

    served = getattr(response, "model", None)
    rest = [m for m in pool if m != served] if isinstance(served, str) else pool[1:]
    rest = rest or pool[1:]
    if not rest:
        logger.warning(
            "%s: blank reply and no other model to try | %s",
            label, _blank_reply_shape(response),
        )
        return response

    logger.warning(
        "%s: blank reply — retrying on %s | %s",
        label, rest[0], _blank_reply_shape(response),
    )
    retry = await _call_with_retry(client, rest, f"{label}-blank-retry", **api_kwargs)
    if is_blank_reply(retry):
        logger.warning(
            "%s: blank reply again after rotating models | %s",
            label, _blank_reply_shape(retry),
        )
    return retry


class BaseAgent(ABC):
    agent_name: str = "base"
    system_prompt: str = "You are a helpful AI assistant for an e-commerce store owner."
    model_override: str | None = None

    def __init__(self):
        self._settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=self._settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            # The openai SDK's default is 600s. A model that's degraded (not
            # outright erroring, just slow) would sit inside a single
            # chat.completions.create() call for up to 10 minutes with no
            # exception raised — _call_with_retry only rotates to the next
            # model on _RETRYABLE errors, so a merely-slow call never rotates
            # away on its own. 30s matches the timeout already used for other
            # external API calls in this codebase (services/allegro_service.py)
            # and turns a stuck/degraded model into an APITimeoutError (which
            # IS retryable) that fails over to the next model in the pool
            # instead of one request eating tens of seconds on its own.
            timeout=30.0,
        )

    def _get_model_pool(self) -> list[str]:
        """Return rotation pool: model_override first, then the rest of the fast
        pool as real fallbacks.

        AllegroAgent sets model_override = settings.gemini_model_fast, which used
        to make this a single-model, zero-redundancy "pool": the old code only
        appended settings.gemini_model_fast if it wasn't already in the pool, and
        for AllegroAgent it always was (same value) — so a rate-limited/erroring
        call on that one model had nothing to rotate to and just re-tried itself
        with growing backoff delays (2s/4s/8s per round, agents.base_agent
        _call_with_retry). The query-performance-by-phase chart's production data
        showed exactly this: AllegroAgent's tool-select/interpret LLM calls (both
        model_override-based) regularly taking tens of seconds despite being
        capped at 400 tokens / reasoning disabled — while Orchestrator's classify
        call, which already got the full multi-model pool, did not. Falling back
        through the rest of model_fast_pool() (same pool Orchestrator uses) fixes
        that without changing which model is tried first under normal conditions.
        """
        if self.model_override:
            pool = [self.model_override]
            pool.extend(m for m in self._settings.model_fast_pool() if m not in pool)
            return pool
        return self._settings.model_pool()

    @abstractmethod
    def _get_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions for this agent."""

    @abstractmethod
    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool call and return its string result."""

    async def run(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
        context: str | None = None,
    ) -> AgentResponse:
        system = self._build_system_prompt(context)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages += list(conversation_history or [])
        messages.append({"role": "user", "content": query})

        tools = self._get_tools()
        model_pool = self._get_model_pool()
        perf = StageTimer(f"{self.agent_name}_agent.run")

        for iteration in range(MAX_ITERATIONS):
            logger.info("[%s] Iteration %d, pool=%s", self.agent_name, iteration + 1, model_pool)

            api_kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": self._settings.gemini_max_tokens,
                # Tool selection and result formatting are both deterministic,
                # data-driven tasks here — no extended reasoning needed. Thinking-
                # capable models (gemini-3.5-flash, 2.5-flash) otherwise spend part
                # of max_tokens on invisible reasoning before the visible answer,
                # which against a large max_tokens budget can add many seconds per
                # call (see agents/orchestrator.py._classify_with_llm for the same
                # fix, and agents/allegro/allegro_agent.py.run for where this was
                # traced as the main cause of slow replies).
                "reasoning_effort": "none",
            }
            if tools:
                api_kwargs["tools"] = tools

            with perf.stage(f"llm_iter{iteration+1}"):
                response = await _call_for_reply(
                    self._client,
                    model_pool,
                    f"{self.agent_name}/iter{iteration+1}",
                    **api_kwargs,
                )
            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                final_text = msg.content or ""
                if tools:
                    asyncio.create_task(
                        analyze_for_tool_gap(
                            client=self._client,
                            model=self._settings.gemini_model_fast,
                            query=query,
                            response=final_text,
                            existing_tool_names=[
                                t["function"]["name"] for t in tools
                            ],
                        )
                    )
                perf.log(iterations=iteration + 1)
                return AgentResponse(
                    text=final_text,
                    agent_type=self.agent_name,
                    metadata={
                        "perf_stages": perf.snapshot(),
                        "perf_total_ms": perf.elapsed_ms(),
                    },
                )

            # Append assistant turn — serialize via model_dump() so Gemini-specific
            # fields like thought_signature (required by thinking models) are preserved.
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump(exclude_none=True) for tc in msg.tool_calls],
            })

            # Execute each tool and append results
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                logger.info("[%s] Calling tool: %s(%s)", self.agent_name, tool_name, tool_input)
                try:
                    with perf.stage(f"tool:{tool_name}"):
                        result_text = await self._execute_tool(tool_name, tool_input)
                except Exception as exc:
                    logger.exception("[%s] Tool %s failed: %s", self.agent_name, tool_name, exc)
                    result_text = "An internal error occurred. Please try again."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

        logger.warning("[%s] Reached max iterations (%d)", self.agent_name, MAX_ITERATIONS)
        perf.log(iterations=MAX_ITERATIONS, result="max_iterations")
        return AgentResponse(
            text="I'm sorry, I couldn't complete the request within the allowed steps.",
            agent_type=self.agent_name,
        )

    def _build_system_prompt(self, context: str | None) -> str:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now_local = datetime.now(ZoneInfo("Europe/Warsaw"))
        parts = [self.system_prompt]
        parts.append(f"Current date and time: {now_local.strftime('%Y-%m-%d %H:%M %Z')}")
        if context:
            parts.append(f"## Relevant context\n{context}")
        parts.append(
            "LANGUAGE RULE: Detect the language of the user's message. "
            "If it is Polish, your ENTIRE response must be in Polish. "
            "If it is English, respond in English. "
            "Never mix languages. Tool results may be in English — ignore that and still reply in the user's language."
        )
        return "\n\n".join(parts)
