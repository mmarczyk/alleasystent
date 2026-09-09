from __future__ import annotations

"""
Query analytics service.

Stores every user query to Redis and provides aggregation + LLM clustering.
All functions are fire-and-forget safe (never raise to callers).
"""

import json
import logging
import re
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
    "allegro":           "Allegro",
    "rag":               "Baza wiedzy",
    "none":              "Chitchat / inne",
    # legacy keys (pre-2D routing)
    "general_knowledge": "Baza wiedzy",
    "chitchat":          "Chitchat / inne",
    # legacy keys (the four-way Allegro split, collapsed to "allegro" — see the
    # routing-model note in agents/orchestrator.py). Kept because both Redis
    # lists here are capped ring buffers, not wiped on deploy: records written
    # by the previous version stay readable in the dashboard until they age out.
    "allegro_orders":    "Zamówienia",
    "allegro_offers":    "Oferty",
    "allegro_messaging": "Wiadomości",
    "allegro_account":   "Konto",
}

_FORMAT_LABELS = {
    "chat":      "chat",
    "table":     "tabela",
    "document":  "dokument",
    "dashboard": "dashboard",
}


def _intent_label(intent: str, tool: str | None = None) -> str:
    """Convert 'source:format' (or legacy flat intent) to a human-readable label.

    `tool` is the first Allegro tool the turn actually called. When present it
    names the query type far more precisely than the source half of `intent`
    can — the source is decided before any tool runs, and since it collapsed to
    a bare "allegro" (see agents/orchestrator.py) it no longer distinguishes
    order questions from offer or billing ones at all. The tool does, and it is
    right by construction rather than by guessing at the seller's wording: "ile
    zapłaciłem prowizji od tego zamówienia" used to be filed under Zamówienia
    because it says "zamówienia", though it is answered from the billing tools.

    This mirrors label_for_perf() below, which has always preferred the tool.
    Records written before this (and turns that ran no tool at all — chitchat,
    RAG, auth prompts, errors) still label off `intent`.
    """
    if tool:
        return _TOOL_LABELS.get(tool, tool.replace("_", " ").capitalize())
    if ":" in intent:
        source, fmt = intent.split(":", 1)
        src_label = _SOURCE_LABELS.get(source, source)
        fmt_label = _FORMAT_LABELS.get(fmt, fmt)
        return f"{src_label} [{fmt_label}]" if fmt != "chat" else src_label
    return _SOURCE_LABELS.get(intent, intent)


# ── Query-performance-by-phase chart ────────────────────────────────────────
# Query-type label is derived from which Allegro tool actually ran — the only
# thing that distinguishes one store query from another now that the routing
# source is a bare "allegro" (e.g. "Nowe zamówienia" vs "Zamówienia": same
# source, different tools — get_new_orders vs get_orders). Used by both
# label_for_perf() and _intent_label(). See agents/allegro/allegro_tools.py
# for the full tool list.
_TOOL_LABELS = {
    "get_new_orders":                  "Nowe zamówienia",
    "get_orders":                      "Zamówienia",
    "get_orders_delivery":             "Zamówienia (dostawa)",
    "get_orders_due_today":            "Zamówienia na dziś",
    "get_order_details":               "Szczegóły zamówienia",
    "calculate_order_profit":          "Zysk z zamówienia",
    "get_orders_pending_invoice":      "Faktury do wystawienia",
    "get_order_invoice_data":          "Dane do faktury",
    "issue_invoice_for_order":         "Wystawianie faktury",
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
    "find_buyer_by_contact":           "Szukanie klienta",
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
    "suggest_message_reminder":        "Przypomnienia o wiadomościach",
    "disable_message_reminder":        "Przypomnienia o wiadomościach",
    "suggest_sales_record_reminder":   "Przypomnienia o ewidencji",
    "disable_sales_record_reminder":   "Przypomnienia o ewidencji",
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
- allegro:{chat|table|document|dashboard}: any live store data — orders, shipping,
  tracking, returns, invoices, product listings, prices, stock levels, messages
  to/from buyers, fees, billing, statistics
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


async def log_query(
    user_id: str,
    text: str,
    intent: str,
    response_len: int,
    tool: str | None = None,
    path: str | None = None,
) -> None:
    """Append a query record to Redis. Non-blocking, never raises.

    `tool` is the first Allegro tool the turn called, and is what the dashboard
    labels the query by — see _intent_label.

    `path` is which branch of the routing classifier decided the turn —
    "keyword", "llm" or "inherited" (agents/orchestrator.py PATH_*). It is the
    field that answers whether the LLM classification call is worth replacing
    with a local model: only the "llm" branch pays for a round-trip, and only
    its queries belong in a training corpus (see export_training_corpus).

    Both are optional so a caller with nothing to report — and every record
    written before these fields existed — still works.
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
                "uid": user_id[:40],
                "text": text[:300],
                "intent": intent,
                "tool": tool,
                "path": path,
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

    # Grouped by (intent, tool), not by intent alone: the tool is what the row
    # is labelled by when the turn called one (see _intent_label), so two turns
    # sharing an intent but running different tools are different query types
    # and must not be summed into one row. Turns with no tool keep tool=None
    # and group exactly as before.
    intent_counts = Counter(
        (q.get("intent", "unknown"), q.get("tool") or None) for q in queries
    )
    total = len(queries)

    # Intent rows with percentage
    intents = []
    for (intent, tool), count in intent_counts.most_common():
        intents.append({
            "intent": intent,
            "tool": tool,
            "label": _intent_label(intent, tool),
            "count": count,
            "pct": round(count / total * 100) if total else 0,
        })

    # Which classifier branch decided each turn. This is the whole point of
    # recording `path`: "llm" is the only branch that costs an LLM round-trip,
    # so its share is what says whether replacing that call with a local
    # classifier would buy anything. Records written before the field existed
    # count as "unknown" rather than being dropped, so the shares stay honest
    # about how much of the window predates the measurement.
    path_counts = Counter(q.get("path") or "unknown" for q in queries)
    paths = [
        {
            "path": p,
            "count": c,
            "pct": round(c / total * 100) if total else 0,
        }
        for p, c in path_counts.most_common()
    ]

    # Recent 30 queries
    recent = [
        {
            "text": q.get("text", ""),
            "intent": q.get("intent", ""),
            "tool": q.get("tool") or None,
            "path": q.get("path") or None,
            "label": _intent_label(q.get("intent", ""), q.get("tool") or None),
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
        "paths": paths,
        "recent": recent,
        "gaps": gaps[:20],
        "queries_sample": [q.get("text", "") for q in queries[:200]],
    }


# ── Training-corpus export ──────────────────────────────────────────────────
# The stored queries are the only source of real routing labels, but they are
# also real seller traffic: buyer names, phone numbers, e-mail addresses, NIPs,
# offer and order IDs. None of that is needed to learn an INTENT — "kto to jest
# 601 220 118?" and "kto to jest 880 197 834?" are the same question — so every
# such value is replaced with a placeholder before the corpus leaves Redis.
#
# Replaced, not deleted: the placeholder keeps the shape of the sentence, which
# is itself the signal. A query that names a phone number routes differently
# from one that doesn't, and blanking the number away entirely would destroy
# exactly the feature a classifier should learn.
#
# Dates are the deliberate exception to "digits are PII": they are not personal
# data and they ARE intent-bearing ("koszty od 2026-09-01 do 2026-09-09" is a
# period question), so they collapse to their own placeholder rather than the
# generic number one, keeping "this query names a date range" learnable.
_RE_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_RE_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b")
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
# Seven or more digits, counted ACROSS separators. A bare "\d{7,}" would miss
# every phone number a human actually types — "880 197 834" and "601 220 118"
# are three-digit groups, each far under any sane run-length threshold, so a
# run-based rule would have published them untouched. Runs on the text only
# after _RE_DATE has already consumed the date-shaped digits above.
_RE_LONG_NUMBER = re.compile(r"\+?\d(?:[\s.\-()]*\d){6,}")
_RE_POSTCODE = re.compile(r"\b\d{2}-\d{3}\b")
# A buyer's Allegro login ("jan_kowalski88", "anna.kowalska88", "sklep-abc").
# Matches a token that starts with a letter and carries a digit or underscore
# somewhere in it — the shape agents/allegro/allegro_tools.py calls
# "_LOOKS_LIKE_LOGIN", applied here WITHOUT that module's requirement that an
# account word precede it. That requirement is right for a tool call, where a
# false positive invents a filter and answers the wrong question; it is wrong
# here, where a miss publishes a real person's account name. Redaction inverts
# the trade: over-redacting costs a little signal ("covid-19" would go too),
# under-redacting leaks, so this leans on recall.
# The lookahead scans over dots and dashes too, so "anna.kowalska88" is caught
# whole. Without that it anchored on the digit-bearing half only and published
# "anna." — a first name, in the clear.
_RE_LOGIN = re.compile(
    r"\b(?=[\w.-]*[\d_])[^\W\d_][\w.-]{3,}\b",
    re.UNICODE,
)


def redact(text: str) -> str:
    """Strip personal data from a query, keeping its shape. See the note above.

    Order matters: the identifier-shaped patterns run first, so a date, phone
    number or e-mail is already a placeholder by the time the broader login
    rules see the text and cannot be re-matched by them.
    """
    text = _RE_UUID.sub("<ID>", text)
    text = _RE_DATE.sub("<DATA>", text)
    text = _RE_EMAIL.sub("<EMAIL>", text)
    text = _RE_POSTCODE.sub("<NUMER>", text)
    text = _RE_LONG_NUMBER.sub("<NUMER>", text)

    # A login with neither digit nor underscore ("z konta sklep-abc") is
    # invisible to the shape rule below — its lookahead requires one, and
    # dropping that requirement would swallow every hyphenated word in the
    # language. The context-anchored extractor catches exactly this case: it
    # keys off the account word in front ("konto", "login", "użytkownik"), so
    # it needs no digit to be sure. Imported lazily to keep this module free of
    # an import-time dependency on the agents package.
    try:
        from agents.allegro.allegro_tools import named_buyer_login
        # A query can name more than one ("z konta a1 albo z konta b2"); each
        # pass removes the leftmost, and the bound stops a pathological input
        # from looping (a replacement that somehow still parses as a login).
        for _ in range(4):
            login = named_buyer_login(text)
            if not login or login not in text:
                break
            text = text.replace(login, "<LOGIN>")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("login redaction unavailable (non-critical): %s", exc)

    return _RE_LOGIN.sub("<LOGIN>", text)


# Only these branches belong in a training corpus. A "keyword" record was
# decided by the regex matcher that still runs IN FRONT of any model, so
# learning from it teaches the model to reproduce a matcher whose answer it
# will never be asked for — no gain, and it drags the training distribution
# towards keyword-shaped queries, which are precisely the ones that never reach
# the branch a local classifier would replace.
_TRAINABLE_PATHS = ("llm", "inherited")


async def export_training_corpus(
    paths: tuple[str, ...] = _TRAINABLE_PATHS,
    redact_pii: bool = True,
) -> list[dict]:
    """The stored queries as intent-classification training rows.

    One row per query: the (redacted) text and the routing label the current
    pipeline gave it. Rows are a DISTILLATION target — the label came from the
    Gemini classifier, so a model trained on this learns to imitate that call,
    which is exactly what replacing it requires.

    `paths` filters to the branches worth learning from (see _TRAINABLE_PATHS);
    pass an empty tuple to export everything, including the keyword branch and
    the records written before the field existed.

    `redact_pii=False` is for local inspection of your own data only — it
    returns raw seller traffic and must not be written anywhere shared.
    """
    queries, _ = await _fetch_all()
    rows = []
    for q in queries:
        path = q.get("path") or "unknown"
        if paths and path not in paths:
            continue
        text = q.get("text", "")
        if not text.strip():
            continue
        intent = q.get("intent", "")
        rows.append({
            "text": redact(text) if redact_pii else text,
            # The routing label is the source half of "<source>:<format>" — the
            # format is decided downstream by the tool and is not this
            # classifier's job to predict.
            "source": intent.split(":", 1)[0] if ":" in intent else intent,
            "tool": q.get("tool") or None,
            "path": path,
            "ts": q.get("ts", 0),
        })
    # Deliberately no user id, not even hashed: this is a single-seller
    # workload, so it would identify rather than group.
    return rows


async def export_to_gcs(bucket: str, prefix: str = "routing-corpus") -> dict:
    """Write the redacted corpus to GCS as one timestamped JSONL object.

    Redis holds the queries in a ring buffer capped at _MAX_QUERIES, so at any
    real traffic level the window is days, not months — anything not copied out
    before it rolls over is gone. Appending a dated object per run turns that
    into an accumulating corpus.

    Returns a summary; never raises (a failed export must not take down the
    endpoint that triggered it).
    """
    rows = await export_training_corpus()
    if not rows:
        return {"written": 0, "reason": "no trainable rows in the current window"}
    # Named for the run, not the data: consecutive runs overlap heavily (the
    # ring buffer still holds what the last run saw), so these objects are
    # append-only snapshots to be deduplicated on read, not disjoint shards.
    name = f"{prefix}/{time.strftime('%Y-%m-%dT%H%M%SZ', time.gmtime())}.jsonl"
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    try:
        # Imported lazily so a deployment without the dependency (or without a
        # bucket configured) still imports this module fine.
        from google.cloud import storage
        client = storage.Client()
        client.bucket(bucket).blob(name).upload_from_string(
            body, content_type="application/x-ndjson",
        )
    except Exception as exc:
        logger.error("analytics.export_to_gcs failed: %s", exc)
        return {"written": 0, "error": str(exc), "object": name}
    logger.info("Exported %d routing rows to gs://%s/%s", len(rows), bucket, name)
    return {"written": len(rows), "object": f"gs://{bucket}/{name}"}


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
