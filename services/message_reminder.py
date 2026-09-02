from __future__ import annotations

"""
Message reminder — periodically checks whether any buyer message is still
unread and nudges the seller about it in chat, then adapts to how they reply.

The sibling of services/invoice_reminder.py, and built the same way: same
Cloud Run/Cloud Scheduler pass (see jobs/order_monitor_service.py), same
Warsaw business-hours gate (7:00-20:00), same per-user cadence starting at 2h
that the seller can push out ("przypomnij za 3 godziny"), same Redis-held
question/answer state consulted by agents.orchestrator.Orchestrator.handle().
Read that module's docstring for why the state cannot live in the chat
history: a proactive nudge has no way to know which conversation thread the
seller will eventually answer from.

Distinct from services/message_monitor.py, which they may well run at the
same time. The MONITOR fires once, at the moment a message ARRIVES, and goes
quiet again even if the message is never read. The REMINDER ignores arrival
entirely and keeps asking for as long as something REMAINS unread, which is
what catches the message that came in overnight and got scrolled past.

Its "yes" action only LISTS the unread threads. That is deliberate: unlike
the invoice reminder, which issues real VAT invoices off a bare "tak", there
is no irreversible action here, so a misread reply costs the seller a list
they did not ask for and nothing more. Replying to a buyer stays an explicit,
separately-confirmed step through the agent's own send_message_to_buyer.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_MONITOR_KIND = "message_reminder"
_STATE_KEY = "allegro:message_reminder:state:{user_id}"
_STATE_TTL = 86400 * 30  # 30 days

_TZ = ZoneInfo("Europe/Warsaw")
_WORK_START_HOUR = 7
_WORK_END_HOUR = 20  # exclusive — last check can fire at 19:xx

_FETCH_LIMIT = 20  # Allegro's /messaging/threads caps `limit` at 20
_DEFAULT_INTERVAL_MINUTES = 120
_MIN_SNOOZE_MINUTES = 5
_MAX_SNOOZE_MINUTES = 60 * 24 * 14  # 2 weeks — sanity cap on a misparsed duration

_STATUS_IDLE = "idle"
_STATUS_AWAITING_RESPONSE = "awaiting_response"
_STATUS_AWAITING_DURATION = "awaiting_duration"

# The follow-up asked when the seller defers without saying for how long.
# A constant because _reminder_owns_reply has to recognize it as one of the
# reminder's OWN questions (see there).
_ASK_DURATION_TEXT = "Jasne — na jak długo mam odłożyć przypomnienie o wiadomościach?"

# Wording unique to this reminder's own outgoing messages: every ask says
# "nieprzeczytaną wiadomość"/"nieprzeczytanych wiadomości", plus the "na jak
# długo?" follow-up above. Matched against the LAST ASSISTANT TURN — never
# against the seller's message — to tell whether the assistant's open question
# really is this one.
# Both branches name messages on purpose — see the matching note in
# services/invoice_reminder.py._OWN_ASK_RE for what a generic duration pattern
# costs when two reminders ask the same shape of question.
_OWN_ASK_RE = re.compile(
    r"nieprzeczytan\w*\s+wiadomo|od[łl]o[żz]y[ćc]\s+przypomnienie\s+o\s+wiadomo",
    re.IGNORECASE,
)

# The only thing that lets a reply be read as an answer to this reminder when
# the assistant's last question was about something else entirely.
_MESSAGE_TOPIC_RE = re.compile(r"wiadomo[śs]", re.IGNORECASE)


def _valid_redis_url(url: str | None) -> bool:
    return bool(url and url.startswith(("redis://", "rediss://", "unix://")))


# ── Enable/disable (mirrors services/invoice_reminder.py) ────────────────────

async def is_monitor_enabled(user_id: str) -> bool:
    """Whether the automatic unread-message reminder is turned on for this user."""
    from services.monitor_state import is_monitor_enabled as _is_enabled
    return await _is_enabled(_MONITOR_KIND, user_id)


async def set_monitor_enabled(user_id: str, enabled: bool) -> None:
    """Turn the automatic unread-message reminder on/off for this user."""
    from services.monitor_state import set_monitor_enabled as _set_enabled
    await _set_enabled(_MONITOR_KIND, user_id, enabled)
    if not enabled:
        await _clear_state(user_id)


# ── Cron entry point ─────────────────────────────────────────────────────────

async def run_once() -> None:
    """Entry point invoked alongside the other monitors — one polling pass
    over every user with the message reminder enabled, then returns."""
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not _valid_redis_url(redis_url):
        logger.info("Message reminder skipped: REDIS_URL not set or has invalid scheme")
        return

    now = datetime.now(_TZ)
    if not (_WORK_START_HOUR <= now.hour < _WORK_END_HOUR):
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
                logger.warning("Message reminder: user=%s skipped: %s", user_id, exc)
    finally:
        await r.aclose()


async def _poll_user(r, user_id: str, now: datetime) -> None:
    from services.allegro_service import AllegroAPIError, AllegroAuthError, AllegroService

    if not await r.exists(f"allegro:tokens:{user_id}"):
        return

    allegro = AllegroService.get_instance(user_id)
    await allegro._load_tokens_from_redis()
    if not allegro._tokens:
        return

    state = await _load_state(r, user_id)
    next_check_at = _parse_dt(state.get("next_check_at")) if state else None
    if next_check_at and now < next_check_at:
        return  # not due yet — respects the (possibly seller-adjusted) cadence

    interval = (state or {}).get("interval_minutes", _DEFAULT_INTERVAL_MINUTES)
    status = (state or {}).get("status", _STATUS_IDLE)
    reminder_count = (state or {}).get("reminder_count", 0)

    try:
        threads = await _fetch_unread(allegro)
    except (AllegroAuthError, AllegroAPIError) as exc:
        logger.warning("Message reminder: Allegro API error user=%s: %s", user_id, exc)
        return

    if not threads:
        # Everything read (possibly answered since the last ask) — go quiet
        # until the next scheduled check.
        await _save_state(
            r, user_id, status=_STATUS_IDLE, thread_ids=[],
            next_check_at=now + timedelta(minutes=interval),
            interval_minutes=interval, reminder_count=0,
        )
        return

    thread_ids = [t.get("id") for t in threads if t.get("id")]

    if status == _STATUS_IDLE:
        await _ask(user_id, threads, again=False, awaiting_duration=False)
        new_status = _STATUS_AWAITING_RESPONSE
        new_reminder_count = 1
    else:
        # Still waiting on a reply from last time — the seller either never
        # answered at all, or was asked "for how long?" and never said.
        await _ask(user_id, threads, again=True, awaiting_duration=(status == _STATUS_AWAITING_DURATION))
        new_status = status
        new_reminder_count = reminder_count + 1

    await _save_state(
        r, user_id, status=new_status, thread_ids=thread_ids,
        next_check_at=now + timedelta(minutes=interval),
        interval_minutes=interval, reminder_count=new_reminder_count,
    )


async def _fetch_unread(allegro) -> list[dict]:
    """The buyer threads that currently have unread messages, newest first."""
    from services.allegro_service import is_thread_unread, thread_last_message_at

    threads = await allegro.get_message_threads(limit=_FETCH_LIMIT)
    unread = [t for t in threads if is_thread_unread(t) and t.get("id")]
    return sorted(unread, key=thread_last_message_at, reverse=True)


# ── Messaging ────────────────────────────────────────────────────────────────

def _unread_phrase(count: int) -> str:
    if count == 1:
        return "1 nieprzeczytaną wiadomość"
    return f"{count} nieprzeczytanych wiadomości"


def _format_dt_pl(iso_str: str) -> str:
    """ISO 8601 (UTC) → 'DD.MM.RRRR, HH:MM' in Warsaw local time.

    Same rendering the agent uses for message threads, so a thread listed by
    the reminder and the same thread listed in a normal chat answer read
    identically.
    """
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(_TZ).strftime("%d.%m.%Y, %H:%M")
    except ValueError:
        return iso_str


def _format_threads(threads: list[dict]) -> str:
    """One line per unread thread. The thread ID stays in the line for the same
    reason the agent keeps it there: send_message_to_buyer needs it, and it is
    read back from this text on the next turn."""
    from services.allegro_service import thread_last_message_at

    return "\n".join(
        f"- **{(t.get('interlocutor') or {}).get('login', '—')}** — "
        f"ostatnia wiadomość: {_format_dt_pl(thread_last_message_at(t))} — "
        f"wątek `{t.get('id')}`"
        for t in threads
    )


def _format_buyers(threads: list[dict]) -> str:
    logins = [(t.get("interlocutor") or {}).get("login") for t in threads]
    shown = [login for login in logins if login][:5]
    if not shown:
        return ""
    more = len(threads) - len(shown)
    return f" (od: {', '.join(shown)}{f' i {more} więcej' if more > 0 else ''})"


async def _ask(user_id: str, threads: list[dict], again: bool, awaiting_duration: bool) -> None:
    phrase = _unread_phrase(len(threads))
    buyers = _format_buyers(threads)

    if not again:
        text = f"💬 Masz {phrase} od kupujących{buyers}.\n\nPokazać je?"
    elif awaiting_duration:
        text = (
            f"💬 Ponownie przypominam o {phrase} od kupujących{buyers} — "
            "na jak długo mam odłożyć przypomnienie o wiadomościach? Albo napisz „pokaż”, "
            "jeśli chcesz je zobaczyć teraz."
        )
    else:
        text = (
            f"💬 Ponownie przypominam: nadal masz {phrase} od kupujących{buyers}. "
            "Pokazać je?"
        )

    await _notify(user_id, chat_text=text)


async def _notify(user_id: str, chat_text: str) -> None:
    """Deliver the reminder as a chat message from the assistant — nothing else.

    Deliberately no OS push and no entry in the notifications panel, matching
    services/invoice_reminder.py: a reminder should read as the assistant
    writing in the chat, not as a system alert. The message MONITOR
    (services/message_monitor.py) is the one that pushes, and it fires on
    arrival rather than on something staying unread, so the two do not
    duplicate each other.

    The message is queued rather than sent, because the assistant has no open
    channel to the app: it lands in whatever conversation the seller has open
    the next time the app polls /push/pending. `dedupe_tag` means an unanswered
    reminder is replaced by the next one instead of stacking up.
    """
    from services.push_service import store_pending_chat

    await store_pending_chat(user_id, chat_text, dedupe_tag=_MONITOR_KIND)


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
    awaiting a reply. Public accessor for the orchestrator."""
    from config.settings import get_settings
    import redis.asyncio as aioredis

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

    Same three-way test as services/invoice_reminder.py._reminder_owns_reply,
    and for the same reason: the assistant asks plenty of its own questions,
    and a bare "tak" answers the one the seller just read rather than a nudge
    from hours ago, possibly in another thread. So this claims a message only
    when the seller can actually have been answering IT:
      - the assistant's last turn was one of this reminder's own questions, or
      - there is no assistant turn in this thread to answer instead (the
        cross-thread case), or
      - the message names messages itself ("pokaż wiadomości").
    Anything else falls through to normal routing.
    """
    if last_assistant_text is None or not last_assistant_text.strip():
        return True
    if _OWN_ASK_RE.search(last_assistant_text):
        return True
    return bool(_MESSAGE_TOPIC_RE.search(text))


async def handle_reply(
    user_id: str, text: str, last_assistant_text: str | None = None,
) -> str | None:
    """Interpret a chat message as a possible reply to an open unread-message
    reminder. Returns the reply text to show the user if it WAS handled as such
    a reply (the caller should show this instead of routing normally), or None
    if there's no open reminder, or the message is unrelated to it (caller
    should fall through to normal routing).
    """
    state = await get_pending_state(user_id)
    if not state:
        return None

    if not _reminder_owns_reply(text, last_assistant_text):
        logger.info(
            "Message reminder: user=%s reply left to normal routing — the assistant's "
            "last turn was a different question",
            user_id,
        )
        return None

    action, minutes = await _classify_reply(text, state, last_assistant_text)

    if action == "unrelated":
        return None

    if action == "show":
        return await _show_all(user_id, state)

    if action == "decline":
        await set_monitor_enabled(user_id, False)
        return (
            "Ok, wyłączyłem automatyczne przypomnienia o nieprzeczytanych wiadomościach. "
            "Możesz je włączyć ponownie w każdej chwili."
        )

    if action == "snooze_duration":
        minutes = max(_MIN_SNOOZE_MINUTES, min(minutes, _MAX_SNOOZE_MINUTES))
        await _set_snooze(user_id, state, minutes)
        return f"Dobrze, przypomnę o wiadomościach za {_format_duration(minutes)}."

    # snooze_unspecified — the seller wants to defer but didn't say for how long
    await _await_duration(user_id, state)
    return _ASK_DURATION_TEXT


async def _show_all(user_id: str, state: dict) -> str:
    """List what is unread RIGHT NOW rather than replaying the thread IDs the
    ask was built from — the seller may have read or answered some of them in
    the meantime, and showing those back would be wrong."""
    from services.allegro_service import AllegroAPIError, AllegroAuthError, AllegroService

    allegro = AllegroService.get_instance(user_id)
    await allegro._load_tokens_from_redis()

    try:
        threads = await _fetch_unread(allegro)
    except (AllegroAuthError, AllegroAPIError) as exc:
        logger.warning("Message reminder: Allegro API error while listing user=%s: %s", user_id, exc)
        return "Nie udało mi się teraz pobrać wiadomości z Allegro. Spróbuj za chwilę."

    await _resolve_state(user_id, state)

    if not threads:
        return "Nie masz już żadnych nieprzeczytanych wiadomości — wygląda na to, że wszystko odczytane. 🙂"

    return (
        f"Masz {_unread_phrase(len(threads))} od kupujących:\n\n"
        + _format_threads(threads)
        + "\n\nChcesz zobaczyć treść którejś z nich albo na nią odpowiedzieć?"
    )


# ── Reply classification (small dedicated LLM call, same shape as
# services/invoice_reminder.py._classify_reply) ─────────────────────────────

_CLASSIFY_SYSTEM_TEMPLATE = """
Jesteś klasyfikatorem odpowiedzi sprzedawcy Allegro na automatyczne przypomnienie
asystenta o nieprzeczytanych wiadomościach od kupujących.

KONTEKST: Asystent zapytał sprzedawcę, czy pokazać mu {count} nieprzeczytanych
wiadomości od kupujących.{extra}

Sklasyfikuj wiadomość sprzedawcy do JEDNEJ z poniższych akcji i odpowiedz DOKŁADNIE
jednym z poniższych formatów — nic więcej, żadnych wyjaśnień:

SHOW
  — sprzedawca chce zobaczyć te wiadomości teraz (np. "tak", "pokaż", "dawaj",
    "ok pokaż je", "jasne", "proszę").
SNOOZE:<minuty>
  — sprzedawca chce odłożyć to na konkretny, dający się policzyć w minutach czas —
    podaj liczbę minut po dwukropku, np.:
      "za 3 godziny" → SNOOZE:180
      "za pół godziny" → SNOOZE:30
      "za 20 minut" → SNOOZE:20
      "jutro rano" / "jutro o 8" → policz minuty do najbliższej godziny 8:00
      "wieczorem" / "po południu" → policz minuty do godziny 18:00 dzisiaj (a jeśli już
        minęła — do 18:00 jutro)
      samo "2 godziny" (bez czasownika) też liczy się jako SNOOZE, gdy sprzedawca
      odpowiada na wcześniejsze pytanie "na jak długo?"
SNOOZE_UNSPECIFIED
  — sprzedawca chce odłożyć to na później, ale NIE podał żadnego konkretnego czasu
    (np. "później", "nie teraz", "poczekaj", "jeszcze nie").
DECLINE
  — sprzedawca chce WYŁĄCZYĆ te automatyczne przypomnienia w ogóle (np. "przestań
    pytać", "wyłącz to", "nie chcę tych przypomnień", "daj mi spokój").
UNRELATED
  — wiadomość NIE jest odpowiedzią na to przypomnienie, tylko dotyczy czegoś zupełnie
    innego (nowe, niepowiązane pytanie/polecenie).

Odpowiedz TYLKO jednym z: SHOW / SNOOZE:<liczba> / SNOOZE_UNSPECIFIED / DECLINE / UNRELATED.
""".strip()

# Second layer under _reminder_owns_reply: even when the reminder may claim the
# message, the assistant's last turn tells the classifier what the seller was
# most likely answering. Truncated because a last turn can be a full listing,
# and only its opening (where the question would be) matters here.
_LAST_TURN_CHARS = 600
_LAST_TURN_TEMPLATE = (
    "OSTATNIA WIADOMOŚĆ ASYSTENTA w tej rozmowie — to na nią sprzedawca "
    "najprawdopodobniej odpowiada:\n\"\"\"{last}\"\"\"\n"
    "Jeśli wiadomość sprzedawcy odpowiada na TO pytanie, a nie na przypomnienie "
    "o nieprzeczytanych wiadomościach (np. asystent zapytał o coś zupełnie innego, "
    "a sprzedawca odpowiedział „tak”), odpowiedz UNRELATED."
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
    count = len(state.get("thread_ids", []))
    extra = _AWAITING_DURATION_HINT if state.get("status") == _STATUS_AWAITING_DURATION else ""
    system = _CLASSIFY_SYSTEM_TEMPLATE.format(count=count, extra=extra)
    now_local = datetime.now(_TZ)
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
            "message_reminder/classify",
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
        logger.warning("Message reminder: classify LLM call failed: %s", exc)
        return "unrelated", 0

    return _parse_classification(raw)


def _parse_classification(raw: str) -> tuple[str, int]:
    if raw.startswith("SHOW"):
        return "show", 0
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
    r, user_id: str, *, status: str, thread_ids: list[str],
    next_check_at: datetime, interval_minutes: int, reminder_count: int,
) -> None:
    payload = {
        "status": status,
        "thread_ids": thread_ids,
        "next_check_at": next_check_at.isoformat(),
        "interval_minutes": interval_minutes,
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
    now = datetime.now(_TZ)

    async def _do(r):
        await _save_state(
            r, user_id, status=_STATUS_IDLE, thread_ids=state.get("thread_ids", []),
            next_check_at=now + timedelta(minutes=minutes),
            interval_minutes=minutes, reminder_count=0,
        )

    await _with_redis(_do)


async def _await_duration(user_id: str, state: dict) -> None:
    next_check_at = _parse_dt(state.get("next_check_at")) or datetime.now(_TZ)

    async def _do(r):
        await _save_state(
            r, user_id, status=_STATUS_AWAITING_DURATION,
            thread_ids=state.get("thread_ids", []),
            next_check_at=next_check_at,
            interval_minutes=state.get("interval_minutes", _DEFAULT_INTERVAL_MINUTES),
            reminder_count=state.get("reminder_count", 0),
        )

    await _with_redis(_do)


async def _resolve_state(user_id: str, state: dict) -> None:
    """Clear the outstanding ask after it's been acted on (shown), keeping the
    current cadence for the next check."""
    now = datetime.now(_TZ)
    interval = state.get("interval_minutes", _DEFAULT_INTERVAL_MINUTES)

    async def _do(r):
        await _save_state(
            r, user_id, status=_STATUS_IDLE, thread_ids=[],
            next_check_at=now + timedelta(minutes=interval),
            interval_minutes=interval, reminder_count=0,
        )

    await _with_redis(_do)


async def _clear_state(user_id: str) -> None:
    async def _do(r):
        await r.delete(_STATE_KEY.format(user_id=user_id))

    await _with_redis(_do)
