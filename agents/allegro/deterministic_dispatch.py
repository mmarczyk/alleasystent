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


# ── zamowienia: get_new_orders ──────────────────────────────────────────────
_ORDERS_TOPIC_RE = re.compile(r"zamów", re.IGNORECASE)
_ORDERS_NEW_SIGNAL_RE = re.compile(r"\bnow[eyąą]\b|\bnew\b|\boczekuj|co\s+nowego", re.IGNORECASE)
_ORDERS_SINGULAR_RE = re.compile(r"ostatni[eaąm]|najnowsz[ea]|\blast\b", re.IGNORECASE)
# Any of these means the query wants more than a bare listing — a specific
# order's details/status/cost (get_order_details, usually chained off a
# listing call this layer can't perform) or a date range (get_orders).
_ORDERS_BAIL_RE = re.compile(
    r"szczegó[łl]|status\b|adres\b|dane\s+do|kiedy\s+wys[łl]a|co\s+si[eę]\s+dzieje|"
    r"koszt(y)?\s+(tego|przy)|faktur|wysy[łl]k[aąi]\s+do\b|kurier|" + _PERIOD_RE.pattern,
    re.IGNORECASE,
)


def _match_get_new_orders(query: str) -> dict | None:
    if _ORDERS_BAIL_RE.search(query):
        return None
    if _is_count_only(query, _ORDERS_TOPIC_RE):
        return {"count_only": True}
    if not _ORDERS_NEW_SIGNAL_RE.search(query):
        return None  # e.g. bare "jakie zamówienia mam" — ambiguous vs get_orders
    if _ORDERS_SINGULAR_RE.search(query):
        return {"limit": 1}
    return {}


# ── zamowienia: get_orders_delivery ────────────────────────────────────────
# "Zamówienia do wysłania" is a preset of the same listing (see
# _ORDERS_PRESETS in AllegroAgent), so resolving it here costs nothing extra
# and keeps the two most common order questions — "nowe" and "do wysłania" —
# on the same, LLM-free path. Deliberately narrow: only wording that names
# the ready-to-ship state outright, never a deadline question ("wysyłka do
# kiedy"), which is a filter this layer can't compute.
_ORDERS_TO_SEND_RE = re.compile(
    r"do\s+wys[łl]ani|do\s+wysy[łl]k|niewys[łl]an|nie\s+wys[łl]an[oey]|"
    r"gotow\w*\s+do\s+wysy[łl]k|czekaj\w*\s+na\s+wysy[łl]k",
    re.IGNORECASE,
)


def _match_get_orders_delivery(query: str) -> dict | None:
    if not (_ORDERS_TOPIC_RE.search(query) and _ORDERS_TO_SEND_RE.search(query)):
        return None
    if _PERIOD_RE.search(query):
        return None  # a date window needs bought_/dispatch_ bounds this layer can't compute
    if _ORDERS_SINGULAR_RE.search(query):
        return None  # "ostatnie do wysłania" — a limit=1 guess isn't worth the risk
    if _is_count_only(query, _ORDERS_TOPIC_RE):
        return {"count_only": True}
    return {}


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
    "zamowienia": [
        ("get_orders_delivery", _match_get_orders_delivery),
        ("get_new_orders", _match_get_new_orders),
    ],
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
