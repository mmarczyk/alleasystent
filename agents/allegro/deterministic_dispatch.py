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

from agents.allegro.allegro_tools import named_buyer_login, named_phone_number

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
    # 'niespakowane' is NOT here: a negation covers every stage other than the
    # one negated (see _negated_stage), and reading it as NOWE hid every order
    # that was already in realizacji but still unpacked.
    r"do\s+spakowania|spakowa[ćc]",
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
    # 'niewysłane' is NOT here — see _negated_stage: it means every order that
    # has not left yet, packed or not, which is an exclusion, not this stage.
    r"\b(?:za|s)pakowan(?:e|ych|ymi|y|a)\b",
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


# ── Negacja etapu ─────────────────────────────────────────────────────────
# "Niewysłane" is not a stage, it is the ABSENCE of one: logically it means
# every order whose status is anything other than SENT — the packed ones AND
# the ones nobody has touched yet. Read as a positive stage it answered with
# READY_FOR_SHIPMENT alone and silently hid the rest, and (worse) the spaced
# spelling 'nie wysłane' fell through to the WYSŁANE pattern, answering the
# exact opposite question. So a negated stage resolves to an EXCLUSION, which
# get_orders takes as exclude_fulfillment_status.
#
# Written either way — 'niewysłane' or 'nie wysłane' — so the compact form is
# split apart first and both spellings then go through one code path.
_COMPACT_NEGATION_RE = re.compile(r"\bnie(?=[a-ząćęłńóśźż]{4,})", re.IGNORECASE)

# What sits between the "nie" and the stage word: nothing ('nie wysłane'), or
# the auxiliary of a passive/perfect form ('nie zostały wysłane', 'nie są
# spakowane'). Anything else ('nie mam nic do wysłania') is not a negated
# stage — the negation there belongs to the verb, not to the stage.
_NEGATION_LEAD_RE = re.compile(
    r"\bnie\s+(?:zosta[łl]\w*\s+|zostan\w*\s+|by[łl]\w*\s+|s[ąa]\s+|jest\s+|"
    r"maj[ąa]\s+|zd[ąa][żz]y\w*\s+|jeszcze\s+)*$",
    re.IGNORECASE,
)

# Statuses a negated stage drops. The parcel-has-left family is one unit: an
# order IN_TRANSIT or PICKED_UP is every bit as "wysłane" as a SENT one, so
# "niewysłane" has to exclude all four or the answer quietly includes parcels
# that are already at the buyer's.
_DISPATCHED_STATUSES = ["SENT", "IN_TRANSIT", "READY_FOR_PICKUP", "PICKED_UP"]
_STAGE_EXCLUDES: dict[str, list[str]] = {
    "shipped":     list(_DISPATCHED_STATUSES),
    "delivered":   ["PICKED_UP"],
    "to_ship":     ["READY_FOR_SHIPMENT", *_DISPATCHED_STATUSES],
    "in_progress": ["PROCESSING"],
    "new":         ["NEW"],
}


def _split_compact_negations(query: str) -> str:
    """'niewysłane' → 'nie wysłane', so both spellings reach _NEGATION_LEAD_RE.

    Two kinds of word must survive intact, and both are recognised by asking
    the stage patterns themselves rather than by a word list:

    * a negation-shaped word that IS a stage's own vocabulary — 'nietknięte'
      (NOWE), 'nieskończone' (W REALIZACJI): a stage pattern matches it from
      the 'nie', so splitting it would destroy the very wording it names;
    * a word that merely starts with those letters — 'niedziela': nothing
      matches after the split either, so there is nothing to negate.
    """
    out: list[str] = []
    last = 0
    for match in _COMPACT_NEGATION_RE.finditer(query):
        if any(pattern.match(query, match.start()) for _, pattern in _ORDER_STAGE_SIGNALS):
            continue
        split = query[:match.end()] + " " + query[match.end():]
        if not any(pattern.match(split, match.end() + 1) for _, pattern in _ORDER_STAGE_SIGNALS):
            continue
        out.append(query[last:match.end()] + " ")
        last = match.end()
    out.append(query[last:])
    return "".join(out)


def _stage_hits(query: str) -> tuple[set[str], set[str]]:
    """(stages named positively, stages named under a negation)."""
    q = _split_compact_negations(query)
    positive: set[str] = set()
    negated: set[str] = set()
    for stage, pattern in _ORDER_STAGE_SIGNALS:
        for match in pattern.finditer(q):
            target = negated if _NEGATION_LEAD_RE.search(q[:match.start()]) else positive
            target.add(stage)
    return positive, negated


def _order_stage(query: str) -> str | None:
    """The single order stage `query` names POSITIVELY, or None when it names
    none, several, or names one under a negation — an ambiguous, mixed or
    negated question is exactly the case the positive matchers must not
    guess at (a negation goes to _negated_stage instead)."""
    hits, negated = _stage_hits(query)
    if negated:
        return None
    # 'do wysłania' is a shipping PLAN, but it shares its stem with the
    # shipped-already wording ('wysłane'), so this one pair co-fires on a
    # query that names only the DO WYSŁANIA stage. Blanking out the plan
    # phrases tells the two cases apart: if a shipped wording is still there
    # afterwards, the query really did name both stages ("które są spakowane,
    # a które już wysłane") and stays ambiguous. Every other overlap is a
    # genuinely mixed question and bails below.
    if hits == {"to_ship", "shipped"}:
        return None if _STATUS_SHIPPED_RE.search(_STATUS_TO_SHIP_RE.sub(" ", query)) else "to_ship"
    return next(iter(hits)) if len(hits) == 1 else None


def _negated_stage(query: str) -> str | None:
    """The single stage `query` NEGATES ('niewysłane', 'nie zostały wysłane'),
    or None. A query that also names a stage positively ("spakowane, ale
    jeszcze nie wysłane") is two questions at once and goes to the LLM."""
    hits, negated = _stage_hits(query)
    if hits or len(negated) != 1:
        return None
    return next(iter(negated))


# ── zamowienia: the follow-up question about ONE already-shown order ───────
# Every matcher below resolves a LISTING of orders, and 'ile' in a listing
# question counts orders ("ile mam nowych zamówień"). But the very same two
# words open a question about the CONTENTS of one order the assistant just
# showed — "ile w tym zamówieniu jest sztuk" — where 'ile' counts items
# inside it. To a stem matcher the two are indistinguishable ('ile' + the
# 'zamów' stem in both), so the count-only branch used to serve the second
# one an order COUNT ("Masz 22 nowe zamówienia"), passed straight through to
# the seller as the finished answer (see _PASSTHROUGH_TOOLS in
# allegro_agent.py — no interpret call gets a chance to notice the mismatch).
#
# This layer can never serve such a follow-up: it needs get_order_details
# with a UUID that only the conversation history holds, and the matchers here
# see the current query alone. So both of its signals are a bail, handing the
# turn to the LLM — which does get the history, and is told to reuse the
# order it already named (see AllegroAgent.system_prompt).

# 1. Anaphora: a demonstrative pointing back at one order ('w tym zamówieniu',
#    'tego ostatniego zamówienia'), or a bare pronoun standing in for it
#    ('ile jest w nim sztuk'). The demonstrative must sit next to the order
#    noun — a lone 'w tym' is period vocabulary ('w tym miesiącu'), which the
#    matchers already bail on for their own reasons.
_ORDER_ANAPHORA_RE = re.compile(
    r"\bt(?:ym|ego|emu|o|amtym|amtego)\s+"
    r"(?:(?:ostatni\w+|samym|samego|wspomnian\w+|pokazan\w+|powy[żz]sz\w+)\s+)?zam[óo]wieni\w*|"
    r"\bw\s+(?:nim|niej)\b|\bz\s+(?:niego|niej)\b",
    re.IGNORECASE,
)

# 2. The unit being counted: a seller asking about parcels to hand over says
#    'ile paczek' (kept in _ORDERS_COUNT_TOPIC_RE on purpose), but 'sztuk',
#    'pozycji', 'produktów' name what sits INSIDE an order, never orders
#    themselves — so on an order-labelled turn they are always the contents
#    question.
_ORDER_CONTENTS_UNIT_RE = re.compile(
    r"\bsztuk\w*\b|\bszt\.?\b|motk[óo]w|\bpozycj\w*|produkt[óo]w|towar[óo]w|przedmiot[óo]w",
    re.IGNORECASE,
)


# An AMOUNT in an order question is a filter no matcher here extracts: every
# one of them resolves stage wording alone, and the listing they return is
# passed straight to the seller (see _PASSTHROUGH_TOOLS), so a dropped "powyżej
# 400 zł" reads as if it had been applied — the whole store's list presented as
# the answer to a filtered question. Bailing hands the turn to the LLM, which
# can pass min_value/max_value to get_orders.
_AMOUNT_RE = re.compile(
    r"\d[\d\s.,]*\s*(?:z[łl]\b|zlot\w*|z[łl]ot\w*|pln\b)|"
    r"\b(?:powy[żz]ej|poni[żz]ej|ponad|co\s+najmniej|przynajmniej|maksymalnie|"
    r"wi[ęe]cej\s+ni[żz]|mniej\s+ni[żz]|dro[żz]sz\w*\s+ni[żz]|ta[ńn]sz\w*\s+ni[żz])\s+\d",
    re.IGNORECASE,
)


def names_an_amount(query: str) -> bool:
    """True when an order question names a money amount — this layer's bail,
    see _AMOUNT_RE."""
    return bool(_AMOUNT_RE.search(query))


def refers_to_one_known_order(query: str) -> bool:
    """True when an order question is really about ONE order already on
    screen — see the comment above for why that is this layer's bail and not
    its match."""
    return bool(_ORDER_ANAPHORA_RE.search(query) or _ORDER_CONTENTS_UNIT_RE.search(query))


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
    if _ORDERS_BAIL_RE.search(query) or _negated_stage(query):
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


def _match_negated_stage(query: str) -> dict | None:
    """"Niewysłane" / "nie zostały wysłane" / "nieodebrane" → get_orders with
    the negated stage's statuses excluded (see _STAGE_EXCLUDES)."""
    if _ORDER_STAGE_BAIL_RE.search(query):
        return None
    stage = _negated_stage(query)
    if stage is None:
        return None
    if _ORDERS_SINGULAR_RE.search(query):
        return None  # "ostatnie niewysłane" — a limit=1 guess isn't worth the risk
    args: dict = {"exclude_fulfillment_status": _STAGE_EXCLUDES[stage]}
    if _is_count_only(query, _ORDERS_COUNT_TOPIC_RE):
        args["count_only"] = True
    return args


# ── zamowienia: get_orders_due_today ───────────────────────────────────────
# The deadline question is the one order question where a date word is not a
# reason to bail: "dzisiaj" IS this tool's default cut-off (end of today, see
# _ORDERS_PRESETS), so the layer can serve it without a clock of its own. Any
# other horizon ("do jutra", "w piątek", a date) needs a computed cut-off and
# still goes to the LLM — hence the today-word requirement rather than a
# general deadline match.
_TODAY_RE = re.compile(r"dzisiaj|dzi[śs]\b|\btoday\b", re.IGNORECASE)
_DISPATCH_INTENT_RE = re.compile(
    r"wys[łl]a[ćc]|wys[łl]ani|wysy[łl]k|nada[ćc]|nadani|termin", re.IGNORECASE,
)


def _match_get_orders_due_today(query: str) -> dict | None:
    if not (_TODAY_RE.search(query) and _DISPATCH_INTENT_RE.search(query)):
        return None
    if _ORDER_DETAIL_INTENT_RE.search(query) or _ORDERS_SINGULAR_RE.search(query):
        return None
    # "ile dziś wysłałem?" is today + shipping words, but it asks about parcels
    # that already LEFT — the WYSŁANE stage, and a period one at that, so it
    # belongs to the LLM, not to a deadline listing. Blank the plan phrases out
    # first (same overlap as in _order_stage): "co mam dziś do wysłania" is not
    # a past-tense question just because it shares the 'wysłan-' stem.
    if _STATUS_SHIPPED_RE.search(_STATUS_TO_SHIP_RE.sub(" ", query)):
        return None
    # Today is the only period this tool resolves on its own; blank it out and
    # anything left ("do jutra", "w tym tygodniu", a date) means a cut-off the
    # layer can't compute.
    if _PERIOD_RE.search(_TODAY_RE.sub(" ", query)):
        return None
    return {"count_only": True} if _is_count_only(query, _ORDERS_COUNT_TOPIC_RE) else {}


# get_orders_due_today first: "co muszę wysłać dzisiaj" also reads as the DO
# WYSŁANIA stage, but the deadline is the narrower, more useful answer.
# get_new_orders last: the stage matchers are the specific ones, and
# _match_get_new_orders' count-only branch fires on a stage-less question.
_ORDERS_MATCHERS: list[tuple[str, Callable[[str], dict | None]]] = [
    ("get_orders_due_today", _match_get_orders_due_today),
    # Negation first: it and the positive matchers are mutually exclusive by
    # construction (_order_stage bails on a negated query), so the order only
    # decides which one gets asked first — and the negated reading is the one
    # a plain stage matcher used to answer with the opposite listing.
    ("get_orders", _match_negated_stage),
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


# ── kupujacy: find_buyer_by_contact (a phone number in the query) ──────────
# The one argument this tool needs for the question it exists for — "czy mam
# klienta z takim nr telefonu +48 880 197 834" — is written out in the query
# itself, and named_phone_number extracts it far more reliably than a model
# retypes a nine-digit number. The narrowness is in that extractor (see its
# comment for why an offer ID or a NIP never reads as a phone); here it is
# enough to bail on the two things this layer never resolves: a period, and a
# question that also asks for something else about the customer.
_BUYER_CONTACT_BAIL_RE = re.compile(
    r"lista|zestawienie|wszystk\w*\s+(klient|kupuj)|"       # the whole population — get_buyers
    r"napisz|wy[śs]lij|odpisz|faktur",                       # a different tool's job
    re.IGNORECASE,
)


def _match_find_buyer_by_contact(query: str) -> dict | None:
    phone = named_phone_number(query)
    if phone is None:
        return None
    if _BUYER_CONTACT_BAIL_RE.search(query) or _PERIOD_RE.search(query):
        return None
    return {"phone": phone}


# ── monitoring: 8 zero-argument UI-action toggles ───────────────────────────
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
_match_message_monitoring = _monitoring_matcher(_MESSAGES_TOPIC_RE, "suggest_message_monitoring", "disable_message_monitoring")
_match_returns_monitoring = _monitoring_matcher(
    re.compile(r"zwrot|reklamacj", re.IGNORECASE), "suggest_returns_monitoring", "disable_returns_monitoring",
)
# The invoice REMINDER (chat nagging) is now the only invoice automation —
# the silent "new order needs an invoice" monitor it used to be confused with
# was removed (see services/invoice_reminder.py and the archive/invoice-
# monitoring branch). So this matcher answers both the reminder wording
# ("przypominaj mi o fakturach") and the generic monitoring/notification
# wording that used to reach the removed toggle ("włącz powiadomienia o
# fakturach") — there is nothing else left for the latter to mean.
# "przypom" rather than "przypomn": the seller says "przypominaj mi o…" at
# least as often as "przypomnienia o…", and the longer stem missed every
# imperative form, sending a perfectly unambiguous toggle request to the LLM.
_REMINDER_WORD_RE = re.compile(
    r"przypom|nagaj|pytaj\s+mnie|dopytuj|monitor|powiad|notyfikacj", re.IGNORECASE,
)


def _match_invoice_reminder(query: str) -> tuple[str, dict] | None:
    if not (_REMINDER_WORD_RE.search(query) and re.search(r"faktur", query, re.IGNORECASE)):
        return None
    if _DISABLE_RE.search(query):
        return "disable_invoice_reminder", {}
    return "suggest_invoice_reminder", {}


# The monthly ewidencja sprzedaży bezrachunkowej (services/
# sales_record_reminder.py). Same shape as the invoice matcher above, and for
# the same reason it insists on a reminder word: "ewidencja" is a topic the
# seller also just asks ABOUT ("co to jest ewidencja bezrachunkowa?", "do
# kiedy mam ją wystawić?"), and answering those with a toggle button instead
# of an answer is the failure this guard exists to avoid.
_SALES_RECORD_TOPIC_RE = re.compile(r"ewidencj|bezrachunkow|bez\s*rachunk", re.IGNORECASE)


def _match_sales_record_reminder(query: str) -> tuple[str, dict] | None:
    if not (_REMINDER_WORD_RE.search(query) and _SALES_RECORD_TOPIC_RE.search(query)):
        return None
    if _DISABLE_RE.search(query):
        return "disable_sales_record_reminder", {}
    return "suggest_sales_record_reminder", {}


_MONITORING_MATCHERS: list[Callable[[str], tuple[str, dict] | None]] = [
    # Before the invoice matcher: "przypominaj mi o ewidencji zamiast o
    # fakturach" names both, and the ewidencja is the more specific topic.
    _match_sales_record_reminder,
    _match_invoice_reminder,
    _match_order_monitoring,
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
    "kupujacy": [("find_buyer_by_contact", _match_find_buyer_by_contact)],
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
    # A named buyer account ("z konta np1988", "od użytkownika kasia.w") is a
    # filter NO matcher here extracts — every one of them resolves arguments
    # from stage/period wording alone. Serving one anyway would answer a
    # question about ONE buyer with the whole store's listing, so this is a
    # bail exactly like the period one: hand it to the LLM, which can pass
    # buyer_login to get_orders.
    if named_buyer_login(query):
        return None
    if "monitoring" in labels:
        return _match_monitoring(query)
    if len(labels) != 1:
        return None
    label = next(iter(labels))
    # Checked once for the whole label rather than inside each order matcher:
    # a follow-up about one already-shown order is unservable HERE whichever
    # listing preset the wording would otherwise resolve to, and a matcher
    # added later inherits the bail instead of having to repeat it.
    if label == "zamowienia" and (refers_to_one_known_order(query) or names_an_amount(query)):
        return None
    for tool_name, matcher in _LABEL_MATCHERS.get(label, []):
        result = matcher(query)
        if result is not None:
            return tool_name, result
    return None
