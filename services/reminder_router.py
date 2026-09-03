from __future__ import annotations

"""
Routes an incoming chat message to the reminder it answers — or, when more
than one could plausibly claim it, asks the seller which one they meant
instead of guessing.

Two reminders write proactively into the chat and then wait for a reply:
services/invoice_reminder.py (unissued VAT invoices) and
services/message_reminder.py (unread buyer messages). Each holds its own
question/answer state in Redis rather than in the conversation, because the
seller may answer from a different chat thread than the nudge landed in — see
those modules' docstrings.

That design leaves one case neither module can settle alone: BOTH are waiting,
the seller types a bare "tak", and there is no assistant turn in this thread to
attribute it to. Whichever module was consulted first used to win the tie
silently. That is the wrong default here — the invoice reminder's "yes" issues
real VAT invoices, which cannot be taken back, and the seller may well have
meant the messages one. So when the reply is genuinely ambiguous this asks
which reminder it belongs to, remembers the original wording, and replays it
against whichever the seller names.

A reply is ambiguous only when nothing identifies its target. Two things
still settle it outright, in this order:
  1. the seller names the topic themselves ("wystaw faktury", "pokaż
     wiadomości") — that reminder takes it even if the other is also open;
  2. the assistant's last turn in this thread was one reminder's own question,
     which is what the seller is visibly answering.
And if the assistant's last turn was some OTHER question entirely, no reminder
claims the reply at all: it falls through to normal routing, where the actual
open question gets answered.
"""

import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Pattern

logger = logging.getLogger(__name__)

_ASK_KEY = "allegro:reminder_router:ask:{user_id}"
# Long enough for the seller to come back to a question they were asked, short
# enough that a forgotten one doesn't hijack an unrelated message hours later —
# by then both reminders have nagged again anyway.
_ASK_TTL = 3600


@dataclass(frozen=True)
class _Spec:
    """What the router needs to know about one reminder."""
    kind: str
    label: str                              # how the disambiguating question names it
    answer_word: str                        # what the seller can type to pick it
    topic_re: Pattern[str]                  # matched against the SELLER's message
    own_ask_re: Pattern[str]                # matched against the LAST ASSISTANT TURN
    get_pending_state: Callable[[str], Awaitable[dict | None]]
    handle_reply: Callable[..., Awaitable[str | None]]


def _specs() -> list[_Spec]:
    """Imported lazily and per call, matching how every other module in this
    package reaches the reminders — they pull in Redis, the Allegro service and
    the OpenAI client, none of which should load at import time."""
    from services import invoice_reminder, message_reminder

    return [
        _Spec(
            kind=invoice_reminder._MONITOR_KIND,
            label="niewystawione faktury",
            answer_word="faktury",
            topic_re=invoice_reminder._INVOICE_TOPIC_RE,
            own_ask_re=invoice_reminder._OWN_ASK_RE,
            get_pending_state=invoice_reminder.get_pending_state,
            handle_reply=invoice_reminder.handle_reply,
        ),
        _Spec(
            kind=message_reminder._MONITOR_KIND,
            label="nieprzeczytane wiadomości",
            answer_word="wiadomości",
            topic_re=message_reminder._MESSAGE_TOPIC_RE,
            own_ask_re=message_reminder._OWN_ASK_RE,
            get_pending_state=message_reminder.get_pending_state,
            handle_reply=message_reminder.handle_reply,
        ),
    ]


def _valid_redis_url(url: str | None) -> bool:
    return bool(url and url.startswith(("redis://", "rediss://", "unix://")))


async def _with_redis(fn):
    import redis.asyncio as aioredis
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not _valid_redis_url(redis_url):
        return None
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        return await fn(r)
    finally:
        await r.aclose()


async def _load_ask(user_id: str) -> dict | None:
    async def _do(r):
        return await r.get(_ASK_KEY.format(user_id=user_id))

    raw = await _with_redis(_do)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def _save_ask(user_id: str, text: str, kinds: list[str]) -> None:
    async def _do(r):
        await r.set(
            _ASK_KEY.format(user_id=user_id),
            json.dumps({"text": text, "kinds": kinds}),
            ex=_ASK_TTL,
        )

    await _with_redis(_do)


async def _clear_ask(user_id: str) -> None:
    async def _do(r):
        await r.delete(_ASK_KEY.format(user_id=user_id))

    await _with_redis(_do)


def _disambiguation_question(specs: list[_Spec]) -> str:
    labels = " czy o ".join(spec.label for spec in specs)
    words = " albo ".join(f"„{spec.answer_word}”" for spec in specs)
    return (
        f"Nie mam pewności, o które przypomnienie chodzi — o {labels}? "
        f"Napisz {words}, żebym wiedział, co zrobić."
    )


async def handle_reply(
    user_id: str, text: str, last_assistant_text: str | None = None,
) -> str | None:
    """Interpret a chat message as a possible reply to an open reminder.

    Returns the text to show the seller if some reminder handled it (or if the
    router needs to ask which one it was for), or None to fall through to
    normal routing.
    """
    specs = _specs()

    # An outstanding "which one did you mean?" takes priority: the seller is
    # most likely answering it right now.
    pending_ask = await _load_ask(user_id)
    if pending_ask:
        wanted = [
            spec for spec in specs
            if spec.kind in pending_ask.get("kinds", []) and spec.topic_re.search(text)
        ]
        await _clear_ask(user_id)
        if len(wanted) == 1:
            spec = wanted[0]
            logger.info("Reminder router: user=%s disambiguated to %s", user_id, spec.kind)
            # Replay what they ORIGINALLY said, not the word they just used to
            # pick — "faktury" on its own is not an instruction, "tak" was.
            # last_assistant_text is dropped with it: the turn that reply was
            # aimed at is two messages back now, and passing the router's own
            # question in its place would only confuse the classifier.
            return await spec.handle_reply(user_id, pending_ask.get("text", text), None)
        # Named both, or neither. Whatever it was, the seller was answering the
        # question and the answer is not usable, so hand the message to normal
        # routing rather than re-asking: "faktury i wiadomości" is something the
        # agent can act on, and re-posing the same question to someone who just
        # tried to answer it is a loop with no exit.
        logger.info("Reminder router: user=%s gave no usable choice, routing normally", user_id)
        return None

    pending = [spec for spec in specs if await spec.get_pending_state(user_id)]
    if not pending:
        return None

    # 1. The seller named the topic — unambiguous whatever else is open.
    named = [spec for spec in pending if spec.topic_re.search(text)]
    if len(named) == 1:
        return await named[0].handle_reply(user_id, text, last_assistant_text)

    # 2. The assistant's last turn in this thread was one reminder's own ask,
    #    so that is visibly what the seller is answering.
    if last_assistant_text and last_assistant_text.strip():
        owned = [spec for spec in pending if spec.own_ask_re.search(last_assistant_text)]
        if len(owned) == 1:
            return await owned[0].handle_reply(user_id, text, last_assistant_text)
        if not owned and not named:
            # The assistant asked something else entirely — this reply belongs
            # to that question, not to a reminder from hours ago.
            return None

    # 3. Nothing identifies the target. With one reminder open that is still
    #    unambiguous; with several, ask rather than guess.
    if len(pending) == 1:
        return await pending[0].handle_reply(user_id, text, last_assistant_text)

    candidates = named or pending
    if len(candidates) == 1:
        return await candidates[0].handle_reply(user_id, text, last_assistant_text)

    logger.info(
        "Reminder router: user=%s ambiguous reply, asking which of %s",
        user_id, [spec.kind for spec in candidates],
    )
    await _save_ask(user_id, text, [spec.kind for spec in candidates])
    return _disambiguation_question(candidates)
