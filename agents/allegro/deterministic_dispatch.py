"""Layer 2/3 of the tool-select pipeline: deterministic (regex-based) tool
+ argument resolution, tried before the tool-select LLM call.

Layer 1 (agents.allegro.allegro_tools.select_tools_for_context) narrows the
tool schemas the LLM sees down to one topic. This module goes one step
further for a curated set of tools whose correct call — both WHICH tool and
WHAT arguments — the query text alone determines with high confidence,
letting AllegroAgent.run() skip the tool-select LLM call entirely for them.
Anything not confidently resolved here returns None and falls through to
the LLM exactly as before, just against the already-shrunk Layer-1 list.

Design principle: a false NEGATIVE here (returning None for a query a human
would find obvious) just costs a normal LLM turn — no worse than before this
module existed. A false POSITIVE — the wrong tool, or the right tool with
wrong/missing arguments — is silently served to the user as ground truth,
which is unacceptable. Every matcher below is deliberately conservative:
- it bails (returns None) on any wording suggesting the query actually needs
  a filter this layer can't safely extract (a named product, a price/date
  range, a specific order) rather than guessing;
- it bails on any wording suggesting a follow-up tool call (chaining) is
  needed, since this layer only ever resolves ONE tool for the whole turn;
- tools whose correct use depends on the LLM matching conversational
  context (e.g. get_thread_messages needing a thread_id inferred from
  earlier turns) are not covered here at all.
"""
from __future__ import annotations

import re
from typing import Callable

# ── Shared building blocks ──────────────────────────────────────────────────
_COUNT_QUESTION_RE = re.compile(r"\b(czy|ile)\b", re.IGNORECASE)
# Any wording that scopes the question to a time window. This layer resolves
# arguments from the query text alone and has no clock, so it can't turn
# "w tym miesiącu" into concrete dates — every matcher must bail on a hit here
# and let the LLM (which is given the current date) fill in the period. Missing
# this is worse than a wrong tool: "ile miałem zwrotów w tym miesiącu" came back
# as the unfiltered all-time count, which reads as a real answer.
_PERIOD_RE = re.compile(
    r"\d{1,2}[./]\d{1,2}|\d{4}-\d{2}-\d{2}|"
    r"stycz|lut[ye]|marc|kwiet|\bmaj\b|czerw|lipc|sierp|wrze[śs]|pa[źz]dzier|listopad|grudni|"
    r"tego\s+(miesi[ąa]ca|tygodnia|roku)|w\s+tym\s+(miesi[ąa]cu|tygodniu|roku)|"
    r"zesz[łl]|ubieg[łl]|poprzedni\w*\s+(miesi[ąa]c\w*|tygodni\w*|rok\w*)|"
    r"ostatni\w*\s+(\d+\s+)?(dni\w*|tygodni\w*|miesi[ąa]c\w*|godzin\w*)|"
    r"dzisiaj|dzi[śs]\b|wczoraj|przedwczoraj|\btoday\b|\bthis\s+month\b|"
    r"od\s+\d|do\s+\d|okres[uie]|zakres",
    re.IGNORECASE,
)
_LIST_OVERRIDE_RE = re.compile(r"poka[żz]|wyświetl|wypisz|\blista\b|jakie\s+(są|mam)|zobacz", re.IGNORECASE)


def _is_count_only(query: str, topic_re: re.Pattern) -> bool:
    """A question word ('czy'/'ile') next to the topic word, with no explicit
    'show me the list' override — mirrors AllegroAgent's own
    _wants_message_count_only, generalized to any topic."""
    if _LIST_OVERRIDE_RE.search(query):
        return False
    return bool(_COUNT_QUESTION_RE.search(query) and topic_re.search(query))


# ── zamowienia: the order-stage vocabulary ──────────────────────────────────
# An order moves through five stages, and a seller names the stage they mean
# in almost every order question they ask — but with wildly different wording
# for the same stage: formal ("gotowe do wysyłki"), colloquial ("co czeka na
# kuriera"), or as a count ("ile paczek do nadania?"). Each bucket below
# collects all three registers for one stage, and each stage maps to the tool
# that can actually filter it:
#
#   nowe          → get_new_orders      (READY_FOR_PROCESSING + NEW)
#   w realizacji  → get_orders          (fulfillment_status=PROCESSING)
#   do wysłania   → get_orders_delivery (READY_FOR_SHIPMENT — its default)
#   wysłane       → get_orders_delivery (fulfillment_status=SENT)
#   odebrane      → get_orders          (fulfillment_status=PICKED_UP)
#
# get_orders WITHOUT a status is deliberately not resolved here: a stage-less
# "pokaż zamówienia" is nearly always really about a period, a buyer or some
# other filter this layer can't extract, so it stays the LLM's fallback.
_ORDERS_TOPIC_RE = re.compile(r"zamów", re.IGNORECASE)
# Counting questions name what the seller physically handles as often as
# they name the order itself — "ile paczek do nadania?", "ile przesyłek
# czeka na kuriera?" — and those must still resolve to count_only.
_ORDERS_COUNT_TOPIC_RE = re.compile(r"zamów|paczk|paczek|przesy[łl]", re.IGNORECASE)
_ORDERS_SINGULAR_RE = re.compile(r"ostatni[eaąm]|najnowsz[ea]|\blast\b", re.IGNORECASE)

# NOWE. "oczekujące" belongs here ("oczekujące na potwierdzenie") EXCEPT when
# it's the awaiting-shipment sense — the trailing \b is load-bearing: without
# it \w* backtracks a letter and the negative lookahead passes on the very
# phrasing it exists to exclude.
_STATUS_NEW_RE = re.compile(
    r"\bnow[eyaąę]\w*|\bnew\b|świe[żz]\w*|swiez\w*|do\s+obs[łl]u[żz]enia|"
    r"z[łl]o[żz]on\w*|zarejestrowan\w*|przyj[ęe]t\w*\s+do\s+systemu|"
    r"oczekuj\w*\b(?!\s+na\s+(?:wysy|wys[łl]a|nada|kurier))|"
    r"co\s+nowego|wpad[łl]\w*|nietkni[ęe]t\w*|na\s+start\w*|do\s+rozpocz[ęe]cia|"
    r"zacz[ąa][ćc]\s+robi[ćc]|w\s+kolejce|"
    # Packing is what turns a NEW order into a READY_FOR_SHIPMENT one, so the
    # TO-DO forms ('do spakowania', 'co mam spakować') are asking for orders
    # that are still NEW — only the DONE forms ('spakowane', 'zapakowane',
    # below) mean the parcel is already waiting for the courier.
    r"do\s+spakowania|spakowa[ćc]|niespakowan\w*",
    re.IGNORECASE,
)

# W REALIZACJI
_STATUS_IN_PROGRESS_RE = re.compile(
    r"w\s+trakcie|w\s+realizacji|na\s+etapie\s+realizacji|przetwarzan\w*|\bw\s+toku\b|"
    r"kompletowan\w*|kompletuj\w*|w\s+robocie|nad\s+czym\s+siedz\w*|"
    r"nie(?:s|u)ko[ńn]czon\w*|do\s+doko[ńn]czenia",
    re.IGNORECASE,
)

# DO WYSŁANIA
_STATUS_TO_SHIP_RE = re.compile(
    r"gotow\w*\s+do\s+(?:wysy[łl]ki|wys[łl]ania|nadania|wyw[óo]zki)|"
    r"do\s+wys[łl]ania|do\s+wysy[łl]ki|do\s+nadania|do\s+wyw[óo]zki|"
    r"(?:o?czekaj\w*|o?czeka|oczekuj\w*)\s+na\s+(?:wysy[łl]k\w*|kuriera|nadanie)|"
    r"na\s+kuriera|przygotowan\w*\s+do\s+nadania|"
    r"\b(?:za|s)pakowan(?:e|ych|ymi|y|a)\b|niewys[łl]an\w*",
    re.IGNORECASE,
)

# WYSŁANE
_STATUS_SHIPPED_RE = re.compile(
    r"wys[łl]an\w*|wys[łl]a[łl]\w*|nadan[eyi]\b|nada[łl]\w*|w\s+transporcie|"
    r"przekazan\w*\s+przewo[źz]nikowi|posz[łl]\w*|wyjecha[łl]\w*|odebra[łl]\s+kurier",
    re.IGNORECASE,
)

# ODEBRANE
_STATUS_DELIVERED_RE = re.compile(
    r"odebran\w*|klient\s+odebra[łl]|dostarczon\w*|zrealizowan\w*|zako[ńn]czon\w*|"
    r"zamkni[ęe]t\w*|potwierdzone\s+odebranie|dotar[łl]\w*|odhaczy\w*",
    re.IGNORECASE,
)

# Stage → (tool, arguments), for the four stages a stage word alone resolves.
# NOWE is absent on purpose: it keeps _match_get_new_orders below, which also
# owns the count-only and singular ("ostatnie zamówienie") branches.
_ORDER_STAGE_TOOLS: dict[str, tuple[str, dict]] = {
    "to_ship":     ("get_orders_delivery", {}),
    "shipped":     ("get_orders_delivery", {"fulfillment_status": "SENT"}),
    "in_progress": ("get_orders", {"fulfillment_status": "PROCESSING"}),
    "delivered":   ("get_orders", {"fulfillment_status": "PICKED_UP"}),
}

_ORDER_STAGE_SIGNALS: tuple[tuple[str, re.Pattern], ...] = (
    ("new", _STATUS_NEW_RE),
    ("in_progress", _STATUS_IN_PROGRESS_RE),
    ("to_ship", _STATUS_TO_SHIP_RE),
    ("shipped", _STATUS_SHIPPED_RE),
    ("delivered", _STATUS_DELIVERED_RE),
)


def _order_stage(query: str) -> str | None:
    """The single order stage `query` names, or None when it names none or
    several — an ambiguous or mixed question is exactly the case this layer
    hands to the LLM rather than guessing at."""
    hits = {stage for stage, pattern in _ORDER_STAGE_SIGNALS if pattern.search(query)}
    # 'do wysłania' / 'niewysłane' are shipping PLANS, but they share their
    # stem with the shipped-already wording ('wysłane'), so this one pair
    # co-fires on a query that names only the DO WYSŁANIA stage. Blanking out
    # the plan phrases tells the two cases apart: if a shipped wording is
    # still there afterwards, the query really did name both stages ("które
    # są spakowane, a które już wysłane") and stays ambiguous. Every other
    # overlap is a genuinely mixed question and bails below.
    if hits == {"to_ship", "shipped"}:
        return None if _STATUS_SHIPPED_RE.search(_STATUS_TO_SHIP_RE.sub(" ", query)) else "to_ship"
    return next(iter(hits)) if len(hits) == 1 else None


# Any of these means the query wants more than a bare listing — a specific
# order's details/status/cost (get_order_details, usually chained off a
# listing call this layer can't perform) or a date range (get_orders).
_ORDERS_BAIL_RE = re.compile(
    r"szczegó[łl]|status\b|adres\b|dane\s+do|kiedy\s+wys[łl]a|co\s+si[eę]\s+dzieje|"
    r"koszt(y)?\s+(tego|przy)|faktur|wysy[łl]k[aąi]\s+do\b|kurier|" + _PERIOD_RE.pattern,
    re.IGNORECASE,
)

# The stage matchers need a narrower bail list than get_new_orders' one above:
# 'kurier' and 'wysyłka do' are core DO WYSŁANIA vocabulary, so bailing on
# them would make those stages unreachable. What stays is what this layer
# genuinely can't serve — one named order, and any period filter.
_ORDER_STAGE_BAIL_RE = re.compile(
    r"szczegó[łl]|status\b|adres\b|dane\s+do|co\s+si[eę]\s+dzieje|koszt(y)?\s+(tego|przy)|"
    r"faktur|[0-9a-f]{8}-[0-9a-f]{4}-|wysy[łl]k[aąi]\s+do\b|termin|"
    r"musz[ęe]\s+wys[łl]a|do\s+kiedy|" + _PERIOD_RE.pattern,
    re.IGNORECASE,
)


def _match_get_new_orders(query: str) -> dict | None:
    if _ORDERS_BAIL_RE.search(query):
        return None
    stage = _order_stage(query)
    if stage is not None and stage != "new":
        return None  # a later stage — one of the stage matchers owns this query
    if _is_count_only(query, _ORDERS_TOPIC_RE):
        return {"count_only": True}
    if stage != "new":
        return None  # e.g. bare "jakie zamówienia mam" — ambiguous vs get_orders
    if _ORDERS_SINGULAR_RE.search(query):
        return {"limit": 1}
    return {}


# All four stages below are presets of the SAME listing call (see
# _ORDERS_PRESETS in AllegroAgent), so resolving them here costs nothing
# beyond the arguments themselves and keeps the most common order questions
# on the LLM-free path.
def _stage_matcher(stage: str) -> Callable[[str], dict | None]:
    def _match(query: str) -> dict | None:
        if _ORDER_STAGE_BAIL_RE.search(query) or _order_stage(query) != stage:
            return None
        if _ORDERS_SINGULAR_RE.search(query):
            return None  # "ostatnie do wysłania" — a limit=1 guess isn't worth the risk
        args = dict(_ORDER_STAGE_TOOLS[stage][1])
        if _is_count_only(query, _ORDERS_COUNT_TOPIC_RE):
            args["count_only"] = True
        return args
    return _match


# get_new_orders last: the stage matchers above are the specific ones, and
# _match_get_new_orders' count-only branch fires on a stage-less question.
_ORDERS_MATCHERS: list[tuple[str, Callable[[str], dict | None]]] = [
    *((tool, _stage_matcher(stage)) for stage, (tool, _) in _ORDER_STAGE_TOOLS.items()),
    ("get_new_orders", _match_get_new_orders),
]


# "Szczegóły/status/koszty/faktura ostatniego (nowego) zamówienia" — the
# two-hop get_new_orders(limit=1) -> get_order_details chain (see
# AllegroAgent._resolve_latest_order_chain). Deliberately narrower than a
# bare _ORDERS_BAIL_RE hit above: requires the singular signal too, so a
# plain listing/date-range/courier question (which also bails from
# _match_get_new_orders) never gets treated as this specific chain.
_ORDER_DETAIL_INTENT_RE = re.compile(
    r"szczegó[łl]|status\b|adres\b|dane\s+do|kiedy\s+wys[łl]a|co\s+si[eę]\s+dzieje|"
    r"koszt(y)?\s+(tego|przy)|faktur",
    re.IGNORECASE,
)


def wants_latest_order_details(query: str) -> bool:
    return bool(
        _ORDERS_TOPIC_RE.search(query)
        and _ORDERS_SINGULAR_RE.search(query)
        and _ORDER_DETAIL_INTENT_RE.search(query)
    )


# ── wiadomosci: get_message_threads (list/count only — never content) ──────
_MESSAGES_TOPIC_RE = re.compile(r"wiadomo", re.IGNORECASE)
_MESSAGES_CONTENT_BAIL_RE = re.compile(
    r"treść|co\s+napisał|co\s+pisze|przeczytaj|t[eę]\s+wiadomo|ta\s+wiadomo",
    re.IGNORECASE,
)


def _match_get_message_threads(query: str) -> dict | None:
    if _MESSAGES_CONTENT_BAIL_RE.search(query):
        return None  # wants message TEXT — get_thread_messages, not covered here
    if not _MESSAGES_TOPIC_RE.search(query):
        return None
    if _is_count_only(query, _MESSAGES_TOPIC_RE):
        return {"count_only": True}
    return {}


# ── konto: get_account_info ─────────────────────────────────────────────────
_ACCOUNT_BARE_RE = re.compile(
    r"^(moje|pokaż|jakie)?\s*(dane\s+)?konto\s*(allegro)?[.?!]*$|"
    r"^(jakie|jaki)\s+mam\s+(konto|profil|subskrypcj\w*)[.?!]*$",
    re.IGNORECASE,
)


def _match_get_account_info(query: str) -> dict | None:
    return {} if _ACCOUNT_BARE_RE.match(query.strip()) else None


# ── oferty: get_offers_summary (bare "podsumowanie/statystyki ofert") ──────
_OFFERS_SUMMARY_RE = re.compile(
    r"^(pokaż\s+)?(podsumowanie|statystyk\w*)\s+(moich\s+)?ofert[.?!]*$",
    re.IGNORECASE,
)


def _match_get_offers_summary(query: str) -> dict | None:
    return {} if _OFFERS_SUMMARY_RE.match(query.strip()) else None


# ── zwroty: get_new_returns / get_returns_to_process / get_new_complaints ──
_RETURNS_TOPIC_RE = re.compile(r"zwrot", re.IGNORECASE)
_COMPLAINTS_TOPIC_RE = re.compile(r"reklamacj", re.IGNORECASE)
_RETURNS_TO_PROCESS_SIGNAL_RE = re.compile(
    r"do\s+obs[łl]u[żz]enia|do\s+rozpatrzenia|czekaj[ąa]c|gotowe\s+do\s+zwrotu",
    re.IGNORECASE,
)


def _match_get_returns_to_process(query: str) -> dict | None:
    if not (_RETURNS_TOPIC_RE.search(query) and _RETURNS_TO_PROCESS_SIGNAL_RE.search(query)):
        return None
    if _PERIOD_RE.search(query):
        return None  # needs date_from_local/date_to_local this layer can't compute
    if _is_count_only(query, _RETURNS_TOPIC_RE):
        return {"count_only": True}
    return {}


def _match_get_new_returns(query: str) -> dict | None:
    if not _RETURNS_TOPIC_RE.search(query) or _RETURNS_TO_PROCESS_SIGNAL_RE.search(query):
        return None
    if _PERIOD_RE.search(query):
        return None  # needs date_from_local/date_to_local this layer can't compute
    if _COMPLAINTS_TOPIC_RE.search(query):
        return None  # names both — let the LLM sort out one vs two calls
    if _is_count_only(query, _RETURNS_TOPIC_RE):
        return {"count_only": True}
    return {}


def _match_get_new_complaints(query: str) -> dict | None:
    if not _COMPLAINTS_TOPIC_RE.search(query) or _RETURNS_TOPIC_RE.search(query):
        return None
    if _PERIOD_RE.search(query):
        return None  # needs date_from_local/date_to_local this layer can't compute
    if _is_count_only(query, _COMPLAINTS_TOPIC_RE):
        return {"count_only": True}
    return {}


# ── monitoring: 10 zero-argument UI-action toggles ──────────────────────────
_ENABLE_RE = re.compile(
    r"w[łl][aą]cz|zacznij|chc[eę]\s+(dostawać|otrzymywać)|w[łl][aą]czy[cć]|powiadamiaj|informuj\s+mnie",
    re.IGNORECASE,
)
_DISABLE_RE = re.compile(r"wy[łl][aą]cz|przesta[nń]|zatrzymaj", re.IGNORECASE)
_MONITORING_WORD_RE = re.compile(r"monitor|powiad|notyfikacj|przypomn", re.IGNORECASE)


def _monitoring_matcher(topic_re: re.Pattern, suggest_tool: str, disable_tool: str) -> Callable[[str], tuple[str, dict] | None]:
    def _match(query: str) -> tuple[str, dict] | None:
        if not (_MONITORING_WORD_RE.search(query) and topic_re.search(query)):
            return None
        if _ENABLE_RE.search(query):
            return suggest_tool, {}
        if _DISABLE_RE.search(query):
            return disable_tool, {}
        return None
    return _match


_match_order_monitoring = _monitoring_matcher(_ORDERS_TOPIC_RE, "suggest_order_monitoring", "disable_order_monitoring")
_match_invoice_monitoring = _monitoring_matcher(re.compile(r"faktur", re.IGNORECASE), "suggest_invoice_monitoring", "disable_invoice_monitoring")
_match_message_monitoring = _monitoring_matcher(_MESSAGES_TOPIC_RE, "suggest_message_monitoring", "disable_message_monitoring")
_match_returns_monitoring = _monitoring_matcher(
    re.compile(r"zwrot|reklamacj", re.IGNORECASE), "suggest_returns_monitoring", "disable_returns_monitoring",
)
# Invoice REMINDER (chat nagging) is a distinct feature from invoice
# MONITORING (silent notifier) — see suggest_invoice_reminder's own
# description — so it needs its own, more specific trigger before the
# generic invoice-monitoring matcher above gets a chance to misfire on it.
_REMINDER_WORD_RE = re.compile(r"przypomn|nagaj|pytaj\s+mnie|dopytuj", re.IGNORECASE)


def _match_invoice_reminder(query: str) -> tuple[str, dict] | None:
    if not (_REMINDER_WORD_RE.search(query) and re.search(r"faktur", query, re.IGNORECASE)):
        return None
    if _DISABLE_RE.search(query):
        return "disable_invoice_reminder", {}
    return "suggest_invoice_reminder", {}


_MONITORING_MATCHERS: list[Callable[[str], tuple[str, dict] | None]] = [
    _match_invoice_reminder,  # before the generic invoice-monitoring matcher
    _match_order_monitoring,
    _match_invoice_monitoring,
    _match_message_monitoring,
    _match_returns_monitoring,
]


def _match_monitoring(query: str) -> tuple[str, dict] | None:
    for matcher in _MONITORING_MATCHERS:
        result = matcher(query)
        if result is not None:
            return result
    return None


# ── Registry: tried in this order for a single-label query ─────────────────
# Single-return matchers (tool name implied by the registry key) plus the
# multi-outcome monitoring matcher (handled separately below).
_LABEL_MATCHERS: dict[str, list[tuple[str, Callable[[str], dict | None]]]] = {
    "zamowienia": _ORDERS_MATCHERS,
    "wiadomosci": [("get_message_threads", _match_get_message_threads)],
    "konto":      [("get_account_info", _match_get_account_info)],
    "oferty":     [("get_offers_summary", _match_get_offers_summary)],
    "zwroty": [
        ("get_returns_to_process", _match_get_returns_to_process),
        ("get_new_returns", _match_get_new_returns),
        ("get_new_complaints", _match_get_new_complaints),
    ],
}


def resolve_deterministic(query: str, labels: set[str]) -> tuple[str, dict] | None:
    """Try to resolve `query` to exactly one tool call without an LLM.

    Only attempted when `labels` (from allegro_tools.matched_labels) names
    exactly one topic — a multi-topic turn may need several tools combined,
    which this layer doesn't attempt. Returns (tool_name, tool_input) on a
    confident match, else None (caller falls through to the LLM).

    "monitoring" is exempt from the single-topic rule: a monitoring toggle
    query always ALSO matches its domain label by design (e.g. "włącz
    monitoring zamówień" contains both "zamów" and "monitor" stems), so
    requiring monitoring to be the ONLY label found would make this branch
    dead code — it's checked first instead, whatever else matched.
    """
    if "monitoring" in labels:
        return _match_monitoring(query)
    if len(labels) != 1:
        return None
    label = next(iter(labels))
    for tool_name, matcher in _LABEL_MATCHERS.get(label, []):
        result = matcher(query)
        if result is not None:
            return tool_name, result
    return None
