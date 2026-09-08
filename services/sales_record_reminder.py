from __future__ import annotations

"""
Sales-record reminder — the monthly nag to issue the "ewidencja sprzedaży
bezrachunkowej" (the record of sales made without an invoice or receipt) for
the PREVIOUS month, which has to be in place by the 5th of the current one.

The third of the chat reminders, next to services/invoice_reminder.py and
services/message_reminder.py, and it shares their plumbing: the same Cloud
Run pass (jobs/order_monitor_service.py), the same queued-chat delivery
(services/push_service.store_pending_chat), the same Redis-held question/
answer state consulted through services/reminder_router.py on every incoming
message. Read the invoice reminder's docstring for why that state cannot live
in the conversation history — a proactive nudge has no way to know which chat
thread the seller will answer from.

What makes this one different is that it runs ENTIRELY ON THE CACHE. The other
two ask Allegro what is outstanding and go quiet by themselves once Allegro
says nothing is; this obligation lives in the seller's accounting, which we
cannot see at all. There is nothing to poll, so nothing is polled: the trigger
is the calendar, the only source of truth for "is it done?" is the seller
saying so, and that answer is a key in Redis (`done_periods`). A user with no
Allegro tokens is therefore still reminded — the reminder never touches the
Allegro API.

Its cadence is the calendar too, not an interval the seller nudges around:

  * a month's asks start on the 1st at 8:00 and are always about the month
    that just ended;
  * on the 1st-3rd it asks twice a day, at 8:00 and 20:00;
  * from the 4th on — the deadline being the 5th — four times a day, at 8:00,
    12:00, 16:00 and 20:00;
  * and it keeps going past the deadline, because a missed one still has to be
    issued, until the seller says it is done ("już wystawiłem", "gotowe").

A confirmation marks THAT PERIOD done and nothing more, so the 1st of the next
month starts the whole thing over for the next month. A month that rolls over
unconfirmed is not forgotten either: it is carried in `overdue` and named in
every later ask, because silently dropping an obligation the seller never
confirmed is the one failure this reminder exists to prevent.
"""

import json
import logging
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_MONITOR_KIND = "sales_record_reminder"
_STATE_KEY = "allegro:sales_record_reminder:state:{user_id}"
# Longer than the siblings' 30 days on purpose: the state carries which months
# the seller has already confirmed, and that must outlive a quiet month rather
# than expire into "never done" and start nagging about a settled period.
_STATE_TTL = 86400 * 400

_TZ = ZoneInfo("Europe/Warsaw")

# The seller's window: first ask of the day at 8:00, last at 20:00.
_FIRST_SLOT_HOUR = 8
_LAST_SLOT_HOUR = 20
# Days 1-3 of the month get the two-a-day cadence, the 4th onwards four a day.
_CALM_UNTIL_DAY = 3
_CALM_SLOT_HOURS = (8, 20)
_URGENT_SLOT_HOURS = (8, 12, 16, 20)
# The statutory deadline this whole reminder is built around.
_DEADLINE_DAY = 5

_MIN_SNOOZE_MINUTES = 5
# A snooze is only ever a pause inside one month — anything longer would push
# the seller past the deadline they asked to be reminded about.
_MAX_SNOOZE_MINUTES = 60 * 24 * 7

_STATUS_IDLE = "idle"
_STATUS_AWAITING_RESPONSE = "awaiting_response"
_STATUS_AWAITING_DURATION = "awaiting_duration"

# The follow-up asked when the seller defers without saying for how long.
# A constant because _reminder_owns_reply has to recognize it as one of this
# reminder's OWN questions (see there).
_ASK_DURATION_TEXT = "Jasne — na jak długo mam odłożyć przypomnienie o ewidencji?"

# Wording unique to this reminder's own outgoing messages: every ask names the
# "ewidencję sprzedaży bezrachunkowej", plus the "na jak długo?" follow-up
# above. Matched against the LAST ASSISTANT TURN — never against the seller's
# message — to tell whether the assistant's open question really is this one.
# Both branches name the ewidencja for the reason spelled out in
# services/invoice_reminder.py._OWN_ASK_RE: a generic duration pattern would
# also match the other two reminders' identically-shaped follow-up and snooze
# the wrong one.
_OWN_ASK_RE = re.compile(
    r"ewidencj\w*\s+sprzeda|od[łl]o[żz]y[ćc]\s+przypomnienie\s+o\s+ewidencj",
    re.IGNORECASE,
)

# The only thing that lets a reply be read as an ewidencja reply when the
# assistant's last question was about something else entirely.
_SALES_RECORD_TOPIC_RE = re.compile(
    r"ewidencj|bezrachunkow|bez\s*rachunk", re.IGNORECASE,
)

# ── "Already done" without asking a model ────────────────────────────────────
#
# The seller's answer is the ONLY thing that can settle this reminder (nothing
# else knows whether the ewidencja exists), and the phrasings they use for it
# are few and fixed: "już wystawiłem", "już wysłane", "już gotowe", "zrobione".
# Recognizing those here rather than in the classifier means the reminder stops
# even when the LLM call fails or the model is having a bad day — the failure
# mode of a missed "done" is nagging someone four times a day about work they
# have finished, which is exactly what makes a reminder get turned off.
#
# A question is never a confirmation ("czy ewidencja jest już gotowa?"), and
# neither is a negation or a "jeszcze" ("jeszcze nie wystawiłem", "nie
# pamiętam, czy już wystawiłem") — those go to the classifier, which can read
# them properly. This path only takes the plain, unambiguous confirmations; the
# cost of being strict here is one LLM call, the cost of being loose is a legal
# obligation quietly dropped.
_DONE_VERB = (
    r"(?:wystawi|zrobi|gotow|wys[łl]a|z[łl]o[żz]y|ogarn|za[łl]atwi|zaopiekowa|"
    r"przygotowa|sko[ńn]czy|zamkn)"
)
_ALREADY = r"(?:ju[żz]|w[łl]a[śs]nie|wcze[śs]niej)"
_ALREADY_DONE_RE = re.compile(
    # "już wystawiłem", "już to wysłałem księgowej"
    rf"\b{_ALREADY}\b[^.!?]{{0,40}}?\b{_DONE_VERB}|"
    # "wystawiłem już", "wysłane już wczoraj" — the same thing, other order
    rf"\b{_DONE_VERB}\w*\b[^.!?]{{0,20}}?\b{_ALREADY}\b|"
    # a bare confirmation as the whole answer: "gotowe", "zrobione", "wysłane"
    r"^\s*(?:gotowe|zrobione|wystawione|wys[łl]ane|za[łl]atwione|ogarni[ęe]te|"
    r"zamkni[ęe]te|zaopiekowane)\b",
    re.IGNORECASE,
)
_NOT_A_CONFIRMATION_RE = re.compile(r"\bnie\b|\bjeszcze\b", re.IGNORECASE)


def _looks_already_done(text: str) -> bool:
    """Whether the seller is plainly saying the ewidencja is taken care of."""
    if "?" in text or _NOT_A_CONFIRMATION_RE.search(text):
        return False
    return bool(_ALREADY_DONE_RE.search(text.strip()))


def _valid_redis_url(url: str | None) -> bool:
    return bool(url and url.startswith(("redis://", "rediss://", "unix://")))


# ── Enable/disable (mirrors services/invoice_reminder.py) ────────────────────

async def is_monitor_enabled(user_id: str) -> bool:
    """Whether the monthly sales-record reminder is turned on for this user."""
    from services.monitor_state import is_monitor_enabled as _is_enabled
    return await _is_enabled(_MONITOR_KIND, user_id)


async def set_monitor_enabled(user_id: str, enabled: bool) -> None:
    """Turn the monthly sales-record reminder on/off for this user.

    Turning it OFF clears the open ask but deliberately keeps nothing else:
    there is no state worth preserving across a disable, and a re-enable should
    start from the current month rather than resume a conversation the seller
    never saw the end of.
    """
    from services.monitor_state import set_monitor_enabled as _set_enabled
    await _set_enabled(_MONITOR_KIND, user_id, enabled)
    if not enabled:
        await _clear_state(user_id)


# ── Calendar ─────────────────────────────────────────────────────────────────

_MONTHS_NOMINATIVE = (
    "styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec",
    "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień",
)
_MONTHS_GENITIVE = (
    "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
    "lipca", "sierpnia", "września", "października", "listopada", "grudnia",
)


def _target_period(now: datetime) -> str:
    """The period the seller owes an ewidencja for right now: always the month
    that has just ended, as "YYYY-MM"."""
    first_of_month = now.date().replace(day=1)
    previous = first_of_month - timedelta(days=1)
    return f"{previous.year:04d}-{previous.month:02d}"


def _period_label(period: str) -> str:
    """"2026-08" → "sierpień 2026" — how an ask names the month."""
    try:
        year, month = (int(part) for part in period.split("-"))
        return f"{_MONTHS_NOMINATIVE[month - 1]} {year}"
    except (ValueError, IndexError):
        return period


def _deadline_phrase(now: datetime) -> str:
    """"5 września" — the deadline in the month the seller is standing in."""
    return f"{_DEADLINE_DAY} {_MONTHS_GENITIVE[now.month - 1]}"


def _days_to_deadline(now: datetime) -> int:
    """Days left until the 5th; negative once it has passed."""
    return (date(now.year, now.month, _DEADLINE_DAY) - now.date()).days


def _next_period_start_phrase(now: datetime) -> str:
    """"1 października" — when the reminder will speak up again about the month
    currently running, after the seller has confirmed this one."""
    first_of_next = (now.date().replace(day=1) + timedelta(days=32)).replace(day=1)
    return f"1 {_MONTHS_GENITIVE[first_of_next.month - 1]}"


def _slot_hours(day: int) -> tuple[int, ...]:
    """The times of day this reminder speaks at, which tighten as the deadline
    approaches: twice a day over the 1st-3rd, four times from the 4th on."""
    return _CALM_SLOT_HOURS if day <= _CALM_UNTIL_DAY else _URGENT_SLOT_HOURS


def _due_slot(now: datetime) -> datetime | None:
    """The most recent scheduled slot at or before `now`, or None if the day's
    first one has not come yet.

    Comparing against the LAST slot rather than counting up from the previous
    ask is what makes the cadence hold across restarts and missed passes: an
    outage from 8:00 to 11:00 delivers the 8:00 ask late instead of losing it,
    and a pass that runs twice inside the same slot only ever asks once.
    """
    hours = [h for h in _slot_hours(now.day) if h <= now.hour]
    if not hours:
        return None
    return now.replace(hour=hours[-1], minute=0, second=0, microsecond=0)


# ── Cron entry point ─────────────────────────────────────────────────────────

async def run_once() -> None:
    """Entry point invoked alongside the other monitors — one pass over every
    user with the sales-record reminder enabled, then returns."""
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not _valid_redis_url(redis_url):
        logger.info("Sales record reminder skipped: REDIS_URL not set or has invalid scheme")
        return

    now = datetime.now(_TZ)
    # The 20:00 slot has to survive a pass that lands at 20:03, so the gate is
    # inclusive of the last slot's hour; _due_slot decides the rest.
    if not (_FIRST_SLOT_HOUR <= now.hour <= _LAST_SLOT_HOUR):
        return

    await _poll_all_users(now)


async def _poll_all_users(now: datetime) -> None:
    import redis.asyncio as aioredis
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not _valid_redis_url(redis_url):
        return
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        enabled_keys = await r.keys(f"allegro:{_MONITOR_KIND}_monitor:enabled:*")
        user_ids = {k.split(":")[3] for k in enabled_keys if k.count(":") >= 3}

        for user_id in user_ids:
            try:
                await _poll_user(r, user_id, now)
            except Exception as exc:
                logger.warning("Sales record reminder: user=%s skipped: %s", user_id, exc)
    finally:
        await r.aclose()


async def _poll_user(r, user_id: str, now: datetime) -> None:
    """Decide whether this user is due an ask right now — off the cache alone.

    Unlike the invoice and message reminders there is no Allegro call here and
    no token check: nothing outside Redis knows anything about the ewidencja,
    so a seller whose Allegro connection has lapsed still gets reminded of an
    obligation that is theirs regardless.
    """
    state = await _load_state(r, user_id) or {}
    period = _target_period(now)

    if period in (state.get("done_periods") or []):
        return  # the seller has already said this month's is out

    snooze_until = _parse_dt(state.get("snooze_until"))
    if snooze_until and now < snooze_until:
        return

    slot = _due_slot(now)
    if slot is None:
        return
    last_ask_at = _parse_dt(state.get("last_ask_at"))
    if last_ask_at and last_ask_at >= slot:
        return  # this slot has already been spoken for

    overdue = _rolled_over_periods(state, period)
    rolled_over = period != state.get("period")
    reminder_count = 0 if rolled_over else state.get("reminder_count", 0)

    await _ask(
        user_id, period, overdue, now,
        again=reminder_count > 0,
        awaiting_duration=(not rolled_over and state.get("status") == _STATUS_AWAITING_DURATION),
    )


    # The ask is outstanding from here on, so the state has to say so — an
    # "already done" that lands on an idle state is not claimed by this
    # reminder at all (get_pending_state) and would fall through to normal
    # routing. The one status worth keeping is the "for how long?" follow-up
    # still waiting for its answer within the same month.
    keeps_duration_question = not rolled_over and state.get("status") == _STATUS_AWAITING_DURATION
    await _save_state(
        r, user_id,
        status=_STATUS_AWAITING_DURATION if keeps_duration_question else _STATUS_AWAITING_RESPONSE,
        period=period, overdue=overdue,
        done_periods=state.get("done_periods") or [],
        last_ask_at=now, snooze_until=None,
        reminder_count=reminder_count + 1,
    )


def _rolled_over_periods(state: dict, period: str) -> list[str]:
    """The earlier months still owed, newest first.

    A month the seller never confirmed does not stop mattering when the
    calendar turns over — it is simply late. So when the target period changes,
    the one we were asking about joins this list unless it was confirmed, and
    every ask from then on names it.
    """
    overdue = list(state.get("overdue") or [])
    previous = state.get("period")
    if (
        previous
        and previous != period
        and previous not in (state.get("done_periods") or [])
        and previous not in overdue
    ):
        overdue.insert(0, previous)
    # Anything the seller has confirmed since (a late "za lipiec już wysłane")
    # drops out, and the list is capped: half a year of unfiled months is a
    # conversation to have, not a longer sentence to read.
    done = set(state.get("done_periods") or [])
    return [p for p in overdue if p not in done][:6]


# ── Messaging ────────────────────────────────────────────────────────────────

def _overdue_phrase(overdue: list[str]) -> str:
    if not overdue:
        return ""
    labels = ", ".join(_period_label(p) for p in overdue)
    if len(overdue) == 1:
        return f" Wciąż nie odhaczyliśmy też ewidencji za {labels}."
    return f" Wciąż nie odhaczyliśmy też ewidencji za: {labels}."


def _deadline_sentence(now: datetime) -> str:
    days_left = _days_to_deadline(now)
    if days_left > 1:
        return f"Termin to {_deadline_phrase(now)} — zostały {days_left} dni."
    if days_left == 1:
        return f"Termin to {_deadline_phrase(now)} — został 1 dzień."
    if days_left == 0:
        return f"Termin to {_deadline_phrase(now)} — czyli dzisiaj."
    overdue_days = -days_left
    day_word = "dzień" if overdue_days == 1 else "dni"
    return f"Termin ({_deadline_phrase(now)}) minął {overdue_days} {day_word} temu."


def _build_ask_text(
    period: str, overdue: list[str], now: datetime, again: bool, awaiting_duration: bool,
) -> str:
    label = _period_label(period)
    head = "📒" if _days_to_deadline(now) >= 0 else "⚠️"
    deadline = _deadline_sentence(now)
    extra = _overdue_phrase(overdue)

    if not again:
        # Deliberately not "nowy miesiąc": the first ask of a period is usually
        # the 1st at 8:00, but it is also what a seller sees when they turn the
        # reminder on mid-month, and greeting them with a new month they are
        # three weeks into reads as a bug.
        return (
            f"{head} Trzeba wystawić ewidencję sprzedaży bezrachunkowej "
            f"za {label}. {deadline}{extra}\n\n"
            "Napisz „już wystawiłem”, kiedy będzie gotowa, a przestanę o tym przypominać."
        )
    if awaiting_duration:
        return (
            f"{head} Ponownie przypominam o ewidencji sprzedaży bezrachunkowej za {label}. "
            f"{deadline}{extra}\n\n"
            "Na jak długo mam odłożyć to przypomnienie? Albo napisz „już wystawiłem”, "
            "jeśli temat jest już zaopiekowany."
        )
    return (
        f"{head} Ponownie przypominam: ewidencja sprzedaży bezrachunkowej za {label} "
        f"wciąż czeka. {deadline}{extra}\n\n"
        "Napisz „już wystawiłem”, kiedy będzie gotowa, a przestanę o tym przypominać."
    )


async def _ask(
    user_id: str, period: str, overdue: list[str], now: datetime,
    again: bool, awaiting_duration: bool,
) -> None:
    await _notify(user_id, _build_ask_text(period, overdue, now, again, awaiting_duration))


async def _notify(user_id: str, chat_text: str) -> None:
    """Deliver the reminder as a chat message from the assistant — nothing else.

    No OS push and no entry in the notifications panel, matching the other two
    reminders: this reads as the assistant writing in the chat, not as a system
    alert. `dedupe_tag` means an unanswered ask is replaced by the next one
    instead of stacking up — four a day would otherwise pile into a wall of
    identical messages by the time the seller opens the app.
    """
    from services.push_service import store_pending_chat

    await store_pending_chat(user_id, chat_text, dedupe_tag=_MONITOR_KIND)


# ── Delivery-time refresh ────────────────────────────────────────────────────

# The dedupe tag this reminder's queued chat messages carry, and the handle the
# delivery path (main._refresh_pending_chats) uses to route one back here.
PENDING_CHAT_TAG = _MONITOR_KIND


async def refresh_pending_message(user_id: str, queued_text: str) -> str | None:
    """Bring a queued ask up to date the moment before the seller reads it.

    An ask states how many days are left until the 5th, and then waits in the
    pending-chat queue for up to a day — so the one written on the 3rd reads
    "zostały 2 dni" when it is opened on the 6th. Rebuilding it here costs
    nothing (there is nothing to fetch, the text is a function of the clock and
    the cached state) and keeps the reminder from being wrong about the one
    number it exists to convey.

    Returns the text to show, or None if the ask is no longer live — the seller
    confirmed it from another device, or turned the reminder off — in which
    case the queued message is dropped rather than delivered.
    """
    state = await get_pending_state(user_id)
    if not state:
        logger.info(
            "Sales record reminder: dropping queued message for user=%s — no open ask", user_id,
        )
        return None

    now = datetime.now(_TZ)
    period = state.get("period") or _target_period(now)
    if period in (state.get("done_periods") or []):
        return None

    return _build_ask_text(
        period,
        state.get("overdue") or [],
        now,
        again=state.get("reminder_count", 0) > 1,
        awaiting_duration=state.get("status") == _STATUS_AWAITING_DURATION,
    )


def _format_duration(minutes: int) -> str:
    if minutes >= 60 * 24 and minutes % (60 * 24) == 0:
        days = minutes // (60 * 24)
        return "1 dzień" if days == 1 else f"{days} dni"
    if minutes >= 60 and minutes % 60 == 0:
        hours = minutes // 60
        if hours == 1:
            return "1 godzinę"
        return f"{hours} godziny" if hours < 5 else f"{hours} godzin"
    return "1 minutę" if minutes == 1 else f"{minutes} minut"


# ── Reply handling (called from the orchestrator on every incoming message) ──

async def get_pending_state(user_id: str) -> dict | None:
    """The open reminder state for this user, or None if there isn't one
    awaiting a reply. Public accessor for the orchestrator/router."""
    import redis.asyncio as aioredis
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not _valid_redis_url(redis_url):
        return None
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        state = await _load_state(r, user_id)
    finally:
        await r.aclose()
    if not state or state.get("status") == _STATUS_IDLE:
        return None
    return state


def _reminder_owns_reply(text: str, last_assistant_text: str | None) -> bool:
    """Whether an open reminder may claim this message at all.

    Same three-way test as the other two reminders (see
    services/invoice_reminder.py._reminder_owns_reply for the incident that put
    it there): the assistant asks plenty of its own questions, and a bare "tak"
    answers the one the seller just read rather than a nudge from hours ago,
    possibly in another thread. So this claims a message only when
      - the assistant's last turn was one of this reminder's own questions, or
      - there is no assistant turn in this thread to answer instead, or
      - the message names the ewidencja itself.
    """
    if last_assistant_text is None or not last_assistant_text.strip():
        return True
    if _OWN_ASK_RE.search(last_assistant_text):
        return True
    return bool(_SALES_RECORD_TOPIC_RE.search(text))


async def handle_reply(
    user_id: str, text: str, last_assistant_text: str | None = None,
) -> str | None:
    """Interpret a chat message as a possible reply to an open sales-record
    reminder. Returns the reply text to show the seller if it WAS handled as
    such a reply, or None to fall through to normal routing.
    """
    state = await get_pending_state(user_id)
    if not state:
        return None

    if not _reminder_owns_reply(text, last_assistant_text):
        logger.info(
            "Sales record reminder: user=%s reply left to normal routing — the assistant's "
            "last turn was a different question",
            user_id,
        )
        return None

    # "Już wystawione" settles this without spending an LLM call — and, more to
    # the point, settles it even when that call would have failed.
    if _looks_already_done(text):
        return await _accept_done(user_id, state)

    action, minutes = await _classify_reply(text, state, last_assistant_text)

    if action == "unrelated":
        return None

    if action == "done":
        return await _accept_done(user_id, state)

    if action == "decline":
        await set_monitor_enabled(user_id, False)
        return (
            "Ok, wyłączyłem przypomnienia o ewidencji sprzedaży bezrachunkowej. "
            "Możesz je włączyć ponownie w każdej chwili."
        )

    if action == "snooze_duration":
        minutes = max(_MIN_SNOOZE_MINUTES, min(minutes, _MAX_SNOOZE_MINUTES))
        await _set_snooze(user_id, state, minutes)
        return f"Dobrze, przypomnę o ewidencji za {_format_duration(minutes)}."

    # snooze_unspecified — the seller wants to defer but didn't say for how long
    await _await_duration(user_id, state)
    return _ASK_DURATION_TEXT


async def _accept_done(user_id: str, state: dict) -> str:
    """Take the seller at their word, because nothing else can settle this.

    There is no Allegro or inFakt record to check the claim against — the
    ewidencja lives in their accounting — so the confirmation IS the source of
    truth, and it is written down as one: the period stops being asked about
    for good, not until the next pass. What it does NOT do is generalise: only
    the months actually named in the ask are marked, so the next month starts
    the reminder again from the 1st.
    """
    now = datetime.now(_TZ)
    period = state.get("period") or _target_period(now)
    settled = [period] + [p for p in (state.get("overdue") or []) if p != period]
    await _mark_done(user_id, state, settled)

    labels = ", ".join(_period_label(p) for p in settled)
    if len(settled) > 1:
        head = f"Super — odhaczam ewidencje sprzedaży bezrachunkowej za: {labels}."
    else:
        head = f"Super — odhaczam ewidencję sprzedaży bezrachunkowej za {labels}."
    return (
        f"{head} Nie będę już o nich przypominać.\n\n"
        f"Odezwę się {_next_period_start_phrase(now)} o 8:00 w sprawie ewidencji "
        f"za {_period_label(_target_period(now.replace(day=1) + timedelta(days=32)))}."
    )


# ── Reply classification (small dedicated LLM call, same shape as
# services/invoice_reminder.py._classify_reply) ─────────────────────────────

_CLASSIFY_SYSTEM_TEMPLATE = """
Jesteś klasyfikatorem odpowiedzi sprzedawcy Allegro na automatyczne comiesięczne
przypomnienie asystenta o ewidencji sprzedaży bezrachunkowej.

KONTEKST: Asystent przypomniał sprzedawcy, że do {deadline} musi wystawić ewidencję
sprzedaży bezrachunkowej za {period}.{extra}

Sklasyfikuj wiadomość sprzedawcy do JEDNEJ z poniższych akcji i odpowiedz DOKŁADNIE
jednym z poniższych formatów — nic więcej, żadnych wyjaśnień:

DONE
  — sprzedawca mówi, że temat jest już zaopiekowany: ewidencja jest wystawiona,
    wysłana, gotowa, przekazana księgowej (np. "już wystawiłem", "już wysłane",
    "już gotowe", "zrobione", "ogarnięte", "księgowa ma", "załatwione").
SNOOZE:<minuty>
  — sprzedawca chce odłożyć przypomnienie na konkretny, dający się policzyć w minutach
    czas — podaj liczbę minut po dwukropku, np.:
      "za 3 godziny" → SNOOZE:180
      "przypomnij jutro" / "jutro rano" → policz minuty do najbliższej godziny 8:00
      "za dwa dni" → SNOOZE:2880
      "wieczorem" → policz minuty do godziny 18:00 dzisiaj (a jeśli już minęła —
        do 18:00 jutro)
      samo "2 godziny" (bez czasownika) też liczy się jako SNOOZE, gdy sprzedawca
      odpowiada na wcześniejsze pytanie "na jak długo?"
SNOOZE_UNSPECIFIED
  — sprzedawca chce odłożyć to na później, ale NIE podał żadnego konkretnego czasu
    (np. "później", "nie teraz", "poczekaj", "jeszcze nie zrobiłem").
DECLINE
  — sprzedawca chce WYŁĄCZYĆ te przypomnienia w ogóle (np. "przestań pytać",
    "wyłącz to przypomnienie", "nie chcę tego", "daj mi spokój z ewidencją").
UNRELATED
  — wiadomość NIE jest odpowiedzią na to przypomnienie, tylko dotyczy czegoś zupełnie
    innego (nowe, niepowiązane pytanie/polecenie), ALBO sprzedawca PYTA o ewidencję
    ("czym to jest?", "do kiedy mam czas?") zamiast potwierdzać, że ją zrobił.

DONE wybieraj TYLKO wtedy, gdy sprzedawca faktycznie stwierdza, że rzecz jest już
zrobiona — "jeszcze nie", "zaraz to zrobię" czy "muszę to zrobić" to NIE jest DONE.

Odpowiedz TYLKO jednym z: DONE / SNOOZE:<liczba> / SNOOZE_UNSPECIFIED / DECLINE / UNRELATED.
""".strip()

# Second layer under _reminder_owns_reply: even when the reminder may claim the
# message, the assistant's last turn tells the classifier what the seller was
# most likely answering. Truncated because only its opening (where the question
# would be) matters here.
_LAST_TURN_CHARS = 600
_LAST_TURN_TEMPLATE = (
    "OSTATNIA WIADOMOŚĆ ASYSTENTA w tej rozmowie — to na nią sprzedawca "
    "najprawdopodobniej odpowiada:\n\"\"\"{last}\"\"\"\n"
    "Jeśli wiadomość sprzedawcy odpowiada na TO pytanie, a nie na przypomnienie "
    "o ewidencji sprzedaży bezrachunkowej (np. asystent zapytał o coś zupełnie "
    "innego, a sprzedawca odpowiedział „tak”), odpowiedz UNRELATED."
)

_AWAITING_DURATION_HINT = (
    " Asystent WŁAŚNIE zapytał sprzedawcę, na jak długo odłożyć przypomnienie — jeśli "
    "odpowiedź to sam czas (np. „2 godziny”, „jutro”), bez czasownika, potraktuj to "
    "jako SNOOZE."
)


async def _classify_reply(
    text: str, state: dict, last_assistant_text: str | None = None,
) -> tuple[str, int]:
    from openai import AsyncOpenAI

    from agents.base_agent import _call_with_retry
    from config.settings import get_settings

    settings = get_settings()
    now_local = datetime.now(_TZ)
    extra = _AWAITING_DURATION_HINT if state.get("status") == _STATUS_AWAITING_DURATION else ""
    system = _CLASSIFY_SYSTEM_TEMPLATE.format(
        deadline=_deadline_phrase(now_local),
        period=_period_label(state.get("period") or _target_period(now_local)),
        extra=extra,
    )
    context_messages = []
    if last_assistant_text and last_assistant_text.strip():
        context_messages.append({
            "role": "user",
            "content": _LAST_TURN_TEMPLATE.format(
                last=last_assistant_text.strip()[:_LAST_TURN_CHARS],
            ),
        })

    client = AsyncOpenAI(
        api_key=settings.google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # See agents/base_agent.py BaseAgent.__init__ for why this matters:
        # without it a degraded-but-not-erroring model can sit inside one call
        # for up to the SDK's 600s default with no rotation.
        timeout=30.0,
    )
    try:
        resp = await _call_with_retry(
            client,
            settings.model_fast_pool(),
            "sales_record_reminder/classify",
            max_tokens=60,
            reasoning_effort="none",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Aktualna data i godzina: {now_local.strftime('%Y-%m-%d %H:%M %A')}"},
                *context_messages,
                {"role": "user", "content": text},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
    except Exception as exc:
        logger.warning("Sales record reminder: classify LLM call failed: %s", exc)
        return "unrelated", 0

    return _parse_classification(raw)


def _parse_classification(raw: str) -> tuple[str, int]:
    if raw.startswith("DONE"):
        return "done", 0
    if raw.startswith("DECLINE"):
        return "decline", 0
    if raw.startswith("SNOOZE_UNSPECIFIED"):
        return "snooze_unspecified", 0
    if raw.startswith("SNOOZE"):
        digits = "".join(ch for ch in raw.split(":", 1)[-1] if ch.isdigit())
        if digits:
            return "snooze_duration", int(digits)
        return "snooze_unspecified", 0
    return "unrelated", 0


# ── State persistence ────────────────────────────────────────────────────────

def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


async def _load_state(r, user_id: str) -> dict | None:
    raw = await r.get(_STATE_KEY.format(user_id=user_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def _save_state(
    r, user_id: str, *, status: str, period: str, overdue: list[str],
    done_periods: list[str], last_ask_at: datetime | None,
    snooze_until: datetime | None, reminder_count: int,
) -> None:
    payload = {
        "status": status,
        "period": period,
        "overdue": overdue,
        # Capped for the same reason the overdue list is: only the recent
        # months can still be asked about, the rest is history.
        "done_periods": done_periods[-24:],
        "last_ask_at": last_ask_at.isoformat() if last_ask_at else None,
        "snooze_until": snooze_until.isoformat() if snooze_until else None,
        "reminder_count": reminder_count,
    }
    await r.set(_STATE_KEY.format(user_id=user_id), json.dumps(payload), ex=_STATE_TTL)


async def _with_redis(fn) -> None:
    """Run `fn(r)` against a fresh Redis connection, closing it afterwards.
    Small helper for the one-off state writes triggered by a chat reply (as
    opposed to the cron pass, which already holds a connection)."""
    import redis.asyncio as aioredis
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not _valid_redis_url(redis_url):
        return
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await fn(r)
    finally:
        await r.aclose()


async def _set_snooze(user_id: str, state: dict, minutes: int) -> None:
    """Hold the asks off until then, without touching the slot schedule itself:
    the cadence belongs to the calendar, a snooze is only a pause inside it."""
    now = datetime.now(_TZ)

    async def _do(r):
        await _save_state(
            r, user_id, status=_STATUS_IDLE,
            period=state.get("period") or _target_period(now),
            overdue=state.get("overdue") or [],
            done_periods=state.get("done_periods") or [],
            last_ask_at=_parse_dt(state.get("last_ask_at")),
            snooze_until=now + timedelta(minutes=minutes),
            reminder_count=state.get("reminder_count", 0),
        )

    await _with_redis(_do)


async def _await_duration(user_id: str, state: dict) -> None:
    now = datetime.now(_TZ)

    async def _do(r):
        await _save_state(
            r, user_id, status=_STATUS_AWAITING_DURATION,
            period=state.get("period") or _target_period(now),
            overdue=state.get("overdue") or [],
            done_periods=state.get("done_periods") or [],
            last_ask_at=_parse_dt(state.get("last_ask_at")),
            snooze_until=None,
            reminder_count=state.get("reminder_count", 0),
        )

    await _with_redis(_do)


async def _mark_done(user_id: str, state: dict, periods: list[str]) -> None:
    """Record the months the seller has confirmed and close the open ask."""
    now = datetime.now(_TZ)
    done = list(state.get("done_periods") or [])
    for period in periods:
        if period not in done:
            done.append(period)

    async def _do(r):
        await _save_state(
            r, user_id, status=_STATUS_IDLE,
            period=state.get("period") or _target_period(now),
            overdue=[p for p in (state.get("overdue") or []) if p not in done],
            done_periods=done,
            last_ask_at=_parse_dt(state.get("last_ask_at")),
            snooze_until=None,
            reminder_count=0,
        )

    await _with_redis(_do)


async def _clear_state(user_id: str) -> None:
    async def _do(r):
        await r.delete(_STATE_KEY.format(user_id=user_id))

    await _with_redis(_do)
