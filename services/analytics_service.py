from __future__ import annotations

"""
Query analytics service.

Stores every user query to Redis and provides aggregation + LLM clustering.
All functions are fire-and-forget safe (never raise to callers).
"""

import json
import logging
import time
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

_QUERY_KEY = "analytics:queries"   # Redis list, LPUSH, capped at _MAX
_GAP_KEY   = "analytics:gaps"      # Redis list of LLM-detected tool gaps
_MAX_QUERIES = 2000
_MAX_GAPS    = 500

_SOURCE_LABELS = {
    "allegro_orders":    "Zamówienia",
    "allegro_offers":    "Oferty",
    "allegro_messaging": "Wiadomości",
    "allegro_account":   "Konto",
    "rag":               "Baza wiedzy",
    "none":              "Chitchat / inne",
    # legacy keys (pre-2D routing)
    "general_knowledge": "Baza wiedzy",
    "chitchat":          "Chitchat / inne",
}

_FORMAT_LABELS = {
    "chat":      "chat",
    "table":     "tabela",
    "document":  "dokument",
    "dashboard": "dashboard",
}


def _intent_label(intent: str) -> str:
    """Convert 'source:format' (or legacy flat intent) to a human-readable label."""
    if ":" in intent:
        source, fmt = intent.split(":", 1)
        src_label = _SOURCE_LABELS.get(source, source)
        fmt_label = _FORMAT_LABELS.get(fmt, fmt)
        return f"{src_label} [{fmt_label}]" if fmt != "chat" else src_label
    return _SOURCE_LABELS.get(intent, intent)

_LLM_SYSTEM = (
    "You are an expert product analyst. Respond ONLY with valid JSON — no markdown fences, "
    "no explanation outside the JSON object."
)

_LLM_PROMPT = """You are analyzing queries sent to an AI assistant for Allegro (Polish e-commerce) store owners.

CURRENTLY HANDLED routing (source:format):
- allegro_orders:{chat|table|document|dashboard}: order data, shipping, tracking, returns, invoices
- allegro_offers:{chat|table|document|dashboard}: product listings, prices, stock levels
- allegro_messaging:{chat|document}: messages to/from buyers
- allegro_account:{chat|table|dashboard}: fees, billing, statistics
- rag:{chat|document}: store FAQ, policies (static knowledge base)
- none:chat: greetings, capability questions, chitchat (no data needed)

LAST {n} USER QUERIES (most recent first):
{queries}

TOOL GAP SUGGESTIONS already detected (suggested_tool → count):
{gaps}

Task: Identify patterns in these queries, especially queries that are NOT well handled.

Return exactly this JSON structure:
{{
  "clusters": [
    {{
      "label": "short Polish label (3-5 words)",
      "count_pct": 15,
      "is_handled": false,
      "examples": ["example query 1", "example query 2", "example query 3"],
      "suggested_agent": "snake_case_agent_name",
      "why_valuable": "one sentence in Polish"
    }}
  ],
  "top_missing": [
    "Feature 1 in Polish",
    "Feature 2 in Polish",
    "Feature 3 in Polish"
  ],
  "summary": "2-3 sentences in Polish describing the biggest opportunity"
}}

Include 4-8 clusters. Sort by count_pct descending. Mark is_handled=true only if the intent fully satisfies the query type."""


def _valid_redis_url(url: str | None) -> bool:
    return bool(url and url.startswith(("redis://", "rediss://", "unix://")))


async def log_query(user_id: str, text: str, intent: str, response_len: int) -> None:
    """Append a query record to Redis. Non-blocking, never raises."""
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
        return
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        try:
            entry = json.dumps({
                "ts": round(time.time()),
                "uid": user_id[:40],
                "text": text[:300],
                "intent": intent,
                "rlen": response_len,
            }, ensure_ascii=False)
            await r.lpush(_QUERY_KEY, entry)
            await r.ltrim(_QUERY_KEY, 0, _MAX_QUERIES - 1)
        finally:
            await r.aclose()
    except Exception as exc:
        logger.debug("analytics.log_query failed (non-critical): %s", exc)


async def log_gap(tool_name: str, description: str, query: str, examples: list[str]) -> None:
    """Append a detected tool gap to Redis. Non-blocking, never raises."""
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
        return
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        try:
            entry = json.dumps({
                "ts": round(time.time()),
                "tool": tool_name,
                "desc": description[:200],
                "query": query[:200],
                "examples": examples[:3],
            }, ensure_ascii=False)
            await r.lpush(_GAP_KEY, entry)
            await r.ltrim(_GAP_KEY, 0, _MAX_GAPS - 1)
        finally:
            await r.aclose()
    except Exception as exc:
        logger.debug("analytics.log_gap failed (non-critical): %s", exc)


async def get_stats() -> dict:
    """Return aggregated stats: intent counts, recent queries, gap summary."""
    queries, gaps_raw = await _fetch_all()

    intent_counts = Counter(q.get("intent", "unknown") for q in queries)
    total = len(queries)

    # Intent rows with percentage
    intents = []
    for intent, count in intent_counts.most_common():
        intents.append({
            "intent": intent,
            "label": _intent_label(intent),
            "count": count,
            "pct": round(count / total * 100) if total else 0,
        })

    # Recent 30 queries
    recent = [
        {
            "text": q.get("text", ""),
            "intent": q.get("intent", ""),
            "label": _intent_label(q.get("intent", "")),
            "ts": q.get("ts", 0),
        }
        for q in queries[:30]
    ]

    # Gap summary: group by tool name
    by_tool: dict[str, dict] = defaultdict(lambda: {"count": 0, "desc": "", "examples": []})
    for g in gaps_raw:
        tool = g.get("tool", "unknown")
        by_tool[tool]["count"] += 1
        by_tool[tool]["desc"] = g.get("desc", "")
        if len(by_tool[tool]["examples"]) < 4:
            q = g.get("query", "")
            if q and q not in by_tool[tool]["examples"]:
                by_tool[tool]["examples"].append(q)

    gaps = sorted(
        [{"tool": k, **v} for k, v in by_tool.items()],
        key=lambda x: x["count"],
        reverse=True,
    )

    return {
        "total": total,
        "intents": intents,
        "recent": recent,
        "gaps": gaps[:20],
        "queries_sample": [q.get("text", "") for q in queries[:200]],
    }


async def analyze_with_llm(client, model_pool: list[str]) -> dict:
    """Run LLM clustering on recent queries. Returns structured insights."""
    from agents.base_agent import _call_with_retry

    queries, gaps_raw = await _fetch_all()
    if not queries:
        return {"error": "Brak zapytań do analizy"}

    # Build gaps summary for prompt
    by_tool: Counter = Counter(g.get("tool", "?") for g in gaps_raw)
    gaps_str = "\n".join(f"  {t}: {c}x" for t, c in by_tool.most_common(10)) or "  (brak)"

    query_texts = "\n".join(
        f"  {i+1}. {q.get('text','')}" for i, q in enumerate(queries[:200])
    )

    prompt = _LLM_PROMPT.format(
        n=min(len(queries), 200),
        queries=query_texts,
        gaps=gaps_str,
    )

    import re
    try:
        resp = await _call_with_retry(
            client,
            model_pool,
            "analytics/llm_cluster",
            max_tokens=2000,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {"error": "LLM zwróciło nieprawidłowy JSON", "raw": raw[:200]}
        return json.loads(match.group())
    except Exception as exc:
        logger.error("analytics.analyze_with_llm failed: %s", exc)
        return {"error": str(exc)}


async def _fetch_all() -> tuple[list[dict], list[dict]]:
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
        return [], []
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        try:
            q_raw, g_raw = await r.lrange(_QUERY_KEY, 0, -1), await r.lrange(_GAP_KEY, 0, -1)
            queries = [json.loads(x) for x in q_raw if x]
            gaps    = [json.loads(x) for x in g_raw if x]
            return queries, gaps
        finally:
            await r.aclose()
    except Exception as exc:
        logger.debug("analytics._fetch_all failed: %s", exc)
        return [], []
