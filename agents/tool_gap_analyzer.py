"""Background analyzer: detects missing tools from user query + response pairs."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_SUGGESTIONS_FILE = Path("data/tool_suggestions.jsonl")

_SYSTEM = (
    "You are an expert at analyzing AI assistant conversations to detect missing capabilities. "
    "Respond ONLY with valid JSON — no markdown, no explanation outside the JSON object."
)

_PROMPT = """Conversation between a user and an Allegro store management AI assistant.

Available tools: {tool_names}

USER QUERY:
{query}

ASSISTANT RESPONSE:
{response}

Did the user ask for something the assistant couldn't fully satisfy due to a missing tool?
Signs of a gap: assistant said it can't do something, apologized for lack of capability, gave only a partial answer, or the query clearly needs a capability not covered by any existing tool.

If a gap exists:
{{"gap_detected": true, "tool_name": "snake_case_name", "description": "one sentence: what the tool should do", "example_queries": ["query 1 in user language", "query 2 in user language"]}}

If no gap:
{{"gap_detected": false}}"""


async def analyze_for_tool_gap(
    client: AsyncOpenAI,
    model: str,
    query: str,
    response: str,
    existing_tool_names: list[str],
) -> None:
    """Fire-and-forget: detect capability gaps, log them, and append to JSONL file."""
    try:
        prompt = _PROMPT.format(
            tool_names=", ".join(existing_tool_names),
            query=query[:600],
            response=response[:1200],
        )
        result = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
        )
        raw = (result.choices[0].message.content or "").strip()

        # Handle potential markdown code fences
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return

        data = json.loads(match.group())
        if not data.get("gap_detected"):
            return

        tool_name = data.get("tool_name", "unknown")
        description = data.get("description", "")
        examples = data.get("example_queries", [])

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:600],
            "response_snippet": response[:400],
            "suggested_tool": tool_name,
            "description": description,
            "example_queries": examples,
        }

        # Always log — visible in Railway dashboard even without persistent volume
        logger.info(
            "TOOL_GAP | tool=%s | %s | query=%r",
            tool_name,
            description,
            query[:120],
        )

        # Also persist to file (useful when a data volume is mounted)
        try:
            _SUGGESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _SUGGESTIONS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.debug("Could not write tool_suggestions.jsonl: %s", exc)

    except Exception as exc:
        logger.debug("Tool gap analysis skipped (non-critical): %s", exc)
