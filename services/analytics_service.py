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
_PERF_KEY  = "analytics:perf"      # Redis list of per-request phase-timing breakdowns
_MAX_QUERIES = 2000
_MAX_GAPS    = 500
_MAX_PERF    = 2000

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


# ── Query-performance-by-phase chart ────────────────────────────────────────
# Query-type label is derived from which Allegro tool actually ran — finer
# grained than the data_source bucket above (e.g. "Nowe zamówienia" vs
# "Zamówienia" are both allegro_orders, but different tools: get_new_orders
# vs get_orders). See agents/allegro/allegro_tools.py for the full tool list.
_TOOL_LABELS = {
    "get_new_orders":                  "Nowe zamówienia",
    "get_orders":                      "Zamówienia",
    "get_orders_delivery":             "Zamówienia (dostawa)",
    "get_orders_due_today":            "Zamówienia na dziś",
    "get_order_details":               "Szczegóły zamówienia",
    "get_orders_pending_invoice":      "Faktury do wystawienia",
    "get_order_invoice_data":          "Dane do faktury",
    "issue_invoice_for_order":         "Wystawianie faktury",
    "unblock_invoice_for_order":       "Odblokowanie faktury",
    "preview_pending_invoices":        "Podgląd faktur",
    "attach_invoice_to_allegro_order": "Załączanie faktury",
    "send_invoice_to_ksef":            "Wysyłka do KSeF",
    "get_active_offers":               "Oferty",
    "get_offer_details":               "Szczegóły oferty",
    "get_offers_summary":              "Podsumowanie ofert",
    "query_offers_by_stock":           "Stany magazynowe",
    "query_offers_by_price":           "Ceny ofert",
    "get_products_to_reorder":         "Uzupełnienie zapasów",
    "update_offer_price":              "Zmiana ceny oferty",
    "update_offer_stock":              "Zmiana stanu oferty",
    "get_message_threads":             "Wiadomości",
    "get_thread_messages":             "Treść wiadomości",
    "send_message_to_buyer":           "Wiadomość do kupującego",
    "get_account_info":                "Konto",
    "get_billing_summary":             "Rozliczenia",
    "get_sales_summary":               "Sprzedaż i zysk",
    "get_buyers":                      "Kupujący",
    "get_new_returns":                 "Zwroty",
    "get_returns_to_process":          "Zwroty do obsłużenia",
    "get_new_complaints":              "Reklamacje",
    "suggest_order_monitoring":        "Monitoring zamówień",
    "disable_order_monitoring":        "Monitoring zamówień",
    "suggest_invoice_reminder":        "Przypomnienia o fakturach",
    "disable_invoice_reminder":        "Przypomnienia o fakturach",
    # Invoice MONITORING was removed (see the archive/invoice-monitoring
    # branch); the two labels stay so historical turns that called it still
    # read as a name instead of a raw tool id.
    "suggest_invoice_monitoring":      "Monitoring faktur",
    "disable_invoice_monitoring":      "Monitoring faktur",
    "suggest_message_monitoring":      "Monitoring wiadomości",
    "disable_message_monitoring":      "Monitoring wiadomości",
    "suggest_returns_monitoring":      "Monitoring zwrotów",
    "disable_returns_monitoring":      "Monitoring zwrotów",
}

# Canonical phase order for the chart, chosen from the actual request
# pipeline: Orchestrator.handle() (session_load, classify, session_save)
# wraps whichever agent handles routing — for Allegro queries that's
# AllegroAgent.run() (auth_check, tool_select_llm, the Allegro API call(s),
# interpret_llm). See agents/perf.py StageTimer, the source of every phase
# name below.
_PHASE_ORDER = [
    "session_load", "classify", "auth_check", "tool_select_llm",
    "allegro_call", "interpret_llm", "session_save",
]
_PHASE_LABELS = {
    "session_load":    "Wczytanie sesji",
    "classify":        "Klasyfikacja intencji",
    "auth_check":      "Sprawdzenie autoryzacji",
    "tool_select_llm": "Wybór narzędzia (LLM)",
    "allegro_call":    "Zapytanie do Allegro",
    "interpret_llm":   "Interpretacja wyniku (LLM)",
    "session_save":    "Zapis odpowiedzi",
}


def label_for_perf(data_source: str, tools: list[str] | None) -> str:
    """Query-type label for the phase-timing chart. Prefers the specific tool
    that ran; falls back to the coarser data-source bucket for turns with no
    tool call (chitchat, rag, auth prompts, errors)."""
    if tools:
        primary = tools[0]
        return _TOOL_LABELS.get(primary, primary.replace("_", " ").capitalize())
    return _SOURCE_LABELS.get(data_source, data_source)


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


async def log_perf(label: str, phases: dict[str, float], total_ms: float, cold: bool = False) -> None:
    """Append one request's phase-timing breakdown to Redis. Non-blocking, never raises.

    `phases` comes straight from agents.perf.StageTimer.snapshot() (raw stage
    names, e.g. "tool:get_new_orders") — bucketing into the chart's canonical
    phases happens in get_perf_stats(), not here, so this stays a dumb log.

    `cold` marks the first request Orchestrator.handle() served since process
    start (see agents/orchestrator.py._mark_request) — a likely Cloud Run
    cold start (--min-instances=0), whose container-boot/import time happens
    entirely before any StageTimer starts and would otherwise silently
    inflate (or go unexplained in) this entry's total_ms.
    """
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
                "label": label,
                "total_ms": round(total_ms, 1),
                "phases": {k: round(v, 1) for k, v in phases.items()},
                "cold": cold,
            }, ensure_ascii=False)
            await r.lpush(_PERF_KEY, entry)
            await r.ltrim(_PERF_KEY, 0, _MAX_PERF - 1)
        finally:
            await r.aclose()
    except Exception as exc:
        logger.debug("analytics.log_perf failed (non-critical): %s", exc)


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


async def get_perf_stats(hours: float | None = None) -> dict:
    """Average phase-timing breakdown (ms) per query-type label, for the
    analytics dashboard's query-performance chart.

    `hours`, when given, restricts the average to entries logged in the last
    `hours` — the stored list otherwise holds up to _MAX_PERF requests with
    no time bound, so right after a latency fix ships the chart would still
    mostly reflect pre-fix requests until enough new ones pushed old entries
    out of the window (which for a low-traffic tool label can take days).
    """
    raw = await _fetch_perf()

    phase_keys = list(_PHASE_ORDER)
    phase_labels = [_PHASE_LABELS[p] for p in phase_keys]
    if hours is not None:
        cutoff = time.time() - hours * 3600
        raw = [e for e in raw if e.get("ts", 0) >= cutoff]
    cold_start = _cold_start_stats(raw)
    if not raw:
        return {"phase_keys": phase_keys, "phase_labels": phase_labels, "series": [], "cold_start": cold_start}

    by_label: dict[str, list[dict]] = defaultdict(list)
    for entry in raw:
        by_label[entry.get("label") or "?"].append(entry)

    series = []
    for label, entries in by_label.items():
        n = len(entries)
        sums: dict[str, float] = defaultdict(float)
        for e in entries:
            for stage_name, ms in (e.get("phases") or {}).items():
                # Which specific Allegro tool ran doesn't matter for phase
                # timing, only that a network round-trip to Allegro happened —
                # collapse every "tool:<name>" stage into one bucket.
                bucket = "allegro_call" if stage_name.startswith("tool:") else stage_name
                sums[bucket] += ms
        series.append({
            "label": label,
            "count": n,
            "avg_total_ms": round(sum(e.get("total_ms", 0.0) for e in entries) / n, 1),
            "phases": {phase: round(sums.get(phase, 0.0) / n, 1) for phase in phase_keys},
        })
    series.sort(key=lambda s: -s["count"])

    return {"phase_keys": phase_keys, "phase_labels": phase_labels, "series": series, "cold_start": cold_start}


def _cold_start_stats(raw: list[dict]) -> dict:
    """Compare cold- vs warm-container requests — a container boot (Cloud Run
    --min-instances=0) happens entirely before any phase in `raw` starts
    timing, so a gap here shows up as a higher total_ms with no matching
    phase to explain it. See log_perf()'s `cold` parameter."""
    cold = [e for e in raw if e.get("cold")]
    warm = [e for e in raw if not e.get("cold")]
    return {
        "cold_count": len(cold),
        "warm_count": len(warm),
        "cold_avg_total_ms": round(sum(e.get("total_ms", 0.0) for e in cold) / len(cold), 1) if cold else None,
        "warm_avg_total_ms": round(sum(e.get("total_ms", 0.0) for e in warm) / len(warm), 1) if warm else None,
    }


async def _fetch_perf() -> list[dict]:
    from config.settings import get_settings
    settings = get_settings()
    if not _valid_redis_url(settings.redis_url):
        return []
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        try:
            raw = await r.lrange(_PERF_KEY, 0, -1)
            return [json.loads(x) for x in raw if x]
        finally:
            await r.aclose()
    except Exception as exc:
        logger.debug("analytics._fetch_perf failed: %s", exc)
        return []
