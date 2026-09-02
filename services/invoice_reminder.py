from __future__ import annotations

"""
Invoice reminder — periodically checks for unissued VAT invoices on already-
shipped Allegro orders and nudges the seller about it in chat, then adapts
to how the seller replies, instead of firing a one-shot notification like
services/order_monitor.py and services/return_complaint_monitor.py do.

Runs as part of the same alleasystent-order-monitor Cloud Run service/Cloud
Scheduler cadence as those monitors (see jobs/order_monitor_service.py),
gated to Warsaw business hours (7:00–20:00) and a per-user cadence that
starts at 2h and changes when the seller asks to be reminded again later
("przypomnij za 4 godziny").

State lives in Redis, not in the chat conversation history: the PWA can have
many separate conversation threads (see PRD §6.1/§7.2) and a proactive
reminder has no way to know which one the seller will eventually reply
from, so — unlike a normal chat turn — this can't rely on Firestore
conversation context. Instead the reminder's own question/answer loop is
tracked here and consulted by agents.orchestrator.Orchestrator.handle() on
every incoming message, regardless of which thread or channel it lands on.

That state says an answer is outstanding, never that a given message IS the
answer — and the assistant asks plenty of its own questions in between
("Pokazać szczegóły?"). So handle_reply() also takes the last assistant turn
of the thread the seller is actually replying in, and hands anything that
answers a DIFFERENT open question back to normal routing untouched
(_reminder_owns_reply). Without that, a "tak" meant for the assistant's own
question issued real, irreversible VAT invoices.

Which orders it nags about is filtered through services/invoice_ledger.py:
Allegro reports an invoice only once its PDF was attached to the order, so
an invoice issued in inFakt and left unattached kept the order on the
"niewystawione" list and the reminder kept asking about it — a second "tak"
then issued a second real invoice for it.

The reminder's TEXT is a separate matter: it is recorded as an assistant turn
in whichever conversation it finally gets delivered to (see
main.py._record_assistant_turns), so the assistant can see what it said and
any follow-up reads as a conversation. The state machine above still cannot
depend on that, for the same reason — the seller may answer from elsewhere.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_MONITOR_KIND = "invoice_reminder"
_STATE_KEY = "allegro:invoice_reminder:state:{user_id}"
_STATE_TTL = 86400 * 30  # 30 days

_TZ = ZoneInfo("Europe/Warsaw")
_WORK_START_HOUR = 7
_WORK_END_HOUR = 20  # exclusive — last check can fire at 19:xx

_DEFAULT_INTERVAL_MINUTES = 120
_MIN_SNOOZE_MINUTES = 5
_MAX_SNOOZE_MINUTES = 60 * 24 * 14  # 2 weeks — sanity cap on a misparsed duration

_STATUS_IDLE = "idle"
_STATUS_AWAITING_RESPONSE = "awaiting_response"
_STATUS_AWAITING_DURATION = "awaiting_duration"

# The follow-up the reminder itself asks when the seller defers without saying
# for how long. A constant because _reminder_owns_reply below has to recognize
# it as one of the reminder's OWN questions (see there).
_ASK_DURATION_TEXT = "Jasne — na jak długo mam odłożyć to przypomnienie?"

# Wording unique to the reminder's own outgoing messages: the three "wystawić
# faktury?" asks (all of which say "niewystawioną fakturę"/"niewystawionych
# faktur") and the "na jak długo?" follow-up above. Matched against the LAST
# ASSISTANT TURN — never against the seller's message — to tell whether the
# assistant's open question really is the invoice one.
_OWN_ASK_RE = re.compile(
    r"niewystawion\w*\s+faktur|na\s+jak\s+d[łl]ugo\s+mam\s+od[łl]o[żz]y[ćc]",
    re.IGNORECASE,
)

# The only thing that lets a reply be read as an invoice reply when the
# assistant's last question was about something else entirely.
_INVOICE_TOPIC_RE = re.compile(r"faktur", re.IGNORECASE)


def _valid_redis_url(url: str | None) -> bool:
    return bool(url and url.startswith(("redis://", "rediss://", "unix://")))


# ── Enable/disable (mirrors services/order_monitor.py, services/monitor_state.py) ──

async def is_monitor_enabled(user_id: str) -> bool:
    """Whether the automatic invoice reminder is turned on for this user."""
    from services.monitor_state import is_monitor_enabled as _is_enabled
    return await _is_enabled(_MONITOR_KIND, user_id)


async def set_monitor_enabled(user_id: str, enabled: bool) -> None:
    """Turn the automatic invoice reminder on/off for this user."""
    from services.monitor_state import set_monitor_enabled as _set_enabled
    await _set_enabled(_MONITOR_KIND, user_id, enabled)
    if not enabled:
        await _clear_state(user_id)


# ── Cron entry point ─────────────────────────────────────────────────────────

async def run_once() -> None:
    """Entry point invoked alongside the other monitors — one polling pass
    over every user with the invoice reminder enabled, then returns."""
    from config.settings import get_settings

    redis_url = get_settings().redis_url
    if not _valid_redis_url(redis_url):
        logger.info("Invoice reminder skipped: REDIS_URL not set or has invalid scheme")
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
                logger.warning("Invoice reminder: user=%s skipped: %s", user_id, exc)
    finally:
        await r.aclose()


async def _poll_user(r, user_id: str, now: datetime) -> None:
    from services import invoice_ledger
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
        orders = await allegro.get_orders_needing_invoice(shipped_only=True)
    except (AllegroAuthError, AllegroAPIError) as exc:
        logger.warning("Invoice reminder: Allegro API error user=%s: %s", user_id, exc)
        return

    order_ids = [o.order_id for o in orders]

    # Drop orders this system has already invoiced. Allegro only reports an
    # invoice once its PDF has been ATTACHED to the order — a separate step
    # that often never happens — so get_orders_needing_invoice keeps handing
    # back orders whose invoice exists in inFakt. Asking about those again is
    # exactly how the same order got invoiced twice: the seller answered
    # "tak" to a reminder that should never have been sent. The ledger is the
    # only place that knows (see services/invoice_ledger.py).
    if order_ids:
        already = await invoice_ledger.already_handled(user_id, order_ids)
        if already:
            logger.info(
                "Invoice reminder: user=%s skipping %d already-issued order(s): %s",
                user_id, len(already), ", ".join(sorted(already)),
            )
            order_ids = [oid for oid in order_ids if oid not in already]

    if not order_ids:
        # Nothing pending (possibly resolved since the last ask) — go quiet
        # until the next scheduled check.
        await _save_state(
            r, user_id, status=_STATUS_IDLE, order_ids=[],
            next_check_at=now + timedelta(minutes=interval),
            interval_minutes=interval, reminder_count=0,
        )
        return

    if status == _STATUS_IDLE:
        await _ask(user_id, order_ids, again=False, awaiting_duration=False)
        new_status = _STATUS_AWAITING_RESPONSE
        new_reminder_count = 1
    else:
        # Still waiting on a reply from last time — the seller either never
        # answered at all, or was asked "for how long?" and never said.
        await _ask(user_id, order_ids, again=True, awaiting_duration=(status == _STATUS_AWAITING_DURATION))
        new_status = status
        new_reminder_count = reminder_count + 1

    await _save_state(
        r, user_id, status=new_status, order_ids=order_ids,
        next_check_at=now + timedelta(minutes=interval),
        interval_minutes=interval, reminder_count=new_reminder_count,
    )


# ── Messaging ────────────────────────────────────────────────────────────────

def _pending_invoice_phrase(count: int) -> str:
    return "1 niewystawioną fakturę" if count == 1 else f"{count} niewystawionych faktur"


def _count_phrase(count: int) -> str:
    return "1 fakturę" if count == 1 else f"{count} faktur"


def _format_order_ids(order_ids: list[str]) -> str:
    shown = ", ".join(f"`{oid}`" for oid in order_ids[:10])
    if len(order_ids) > 10:
        shown += f" i {len(order_ids) - 10} więcej"
    return shown


async def _ask(user_id: str, order_ids: list[str], again: bool, awaiting_duration: bool) -> None:
    phrase = _pending_invoice_phrase(len(order_ids))
    ids_str = _format_order_ids(order_ids)

    if not again:
        text = (
            f"🧾 Masz {phrase} dla już wysłanych zamówień: {ids_str}.\n\n"
            "Wystawić je teraz?"
        )
    elif awaiting_duration:
        text = (
            f"🧾 Ponownie Ci przypominam o {phrase} do wystawienia ({ids_str}) — "
            "na jak długo mam odłożyć to przypomnienie? Albo napisz „wystaw”, "
            "jeśli chcesz zrobić to teraz."
        )
    else:
        text = (
            f"🧾 Ponownie Ci przypominam: nadal masz {phrase} dla wysłanych zamówień "
            f"({ids_str}). Wystawić je teraz?"
        )

    await _notify(user_id, chat_text=text)


async def _notify(user_id: str, chat_text: str) -> None:
    """Deliver the reminder as a chat message from the assistant — nothing else.

    Deliberately no OS push and no entry in the notifications panel: the seller
    asked for this reminder to read as the assistant writing to them in the
    chat, the way it would answer a question, rather than as a system alert
    they have to tap through to find. The other monitors (orders, messages,
    returns) are unchanged and still notify.

    The message is queued rather than sent, because the assistant has no open
    channel to the app: it lands in whatever conversation the seller has open
    the next time the app polls /push/pending (startup, resume, or its periodic
    check). `dedupe_tag` means an unanswered reminder is replaced by the next
    one instead of stacking up.
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

    An open reminder is NOT the only thing the assistant may be waiting on an
    answer to. The assistant asks its own questions in the chat ("Masz 1 nową
    wiadomość (od: X). Pokazać szczegóły?"), and a bare "tak" answers the
    question the seller just read — not a reminder they may have been shown
    hours earlier, possibly in another thread. Letting the reminder take that
    "tak" issued real VAT invoices in response to a question about messages,
    which is unacceptable and irreversible.

    So the reminder only claims a message when the seller can actually have
    been answering IT:
      - the assistant's last turn was one of the reminder's own questions, or
      - there is no assistant turn in this thread to answer instead (the
        cross-thread case this module is built for — see the module
        docstring), or
      - the message names invoices itself ("wystaw faktury", "przypomnij mi
        o fakturach jutro"), which is unambiguous whatever was asked before.
    Anything else falls through to normal routing, where the actual open
    question gets answered.
    """
    if last_assistant_text is None or not last_assistant_text.strip():
        return True
    if _OWN_ASK_RE.search(last_assistant_text):
        return True
    return bool(_INVOICE_TOPIC_RE.search(text))


async def handle_reply(
    user_id: str, text: str, last_assistant_text: str | None = None,
) -> str | None:
    """Interpret a chat message as a possible reply to an open invoice
    reminder. Returns the reply text to show the user if it WAS handled as
    such a reply (the caller should show this instead of routing normally),
    or None if there's no open reminder, or the message is unrelated to it
    (caller should fall through to normal routing).

    `last_assistant_text` is the last thing the assistant said in the thread
    the seller is replying in, when the caller can supply it. It decides
    whether this reminder gets to read the message at all
    (_reminder_owns_reply) and is given to the classifier as context.
    """
    state = await get_pending_state(user_id)
    if not state:
        return None

    if not _reminder_owns_reply(text, last_assistant_text):
        logger.info(
            "Invoice reminder: user=%s reply left to normal routing — the assistant's "
            "last turn was a different question",
            user_id,
        )
        return None

    action, minutes = await _classify_reply(text, state, last_assistant_text)

    if action == "unrelated":
        return None

    if action == "issue":
        return await _issue_all(user_id, state)

    if action == "decline":
        await set_monitor_enabled(user_id, False)
        return (
            "Ok, wyłączyłem automatyczne przypomnienia o niewystawionych fakturach. "
            "Możesz je włączyć ponownie w każdej chwili."
        )

    if action == "snooze_duration":
        minutes = max(_MIN_SNOOZE_MINUTES, min(minutes, _MAX_SNOOZE_MINUTES))
        await _set_snooze(user_id, state, minutes)
        return f"Dobrze, przypomnę o fakturach za {_format_duration(minutes)}."

    # snooze_unspecified — the seller wants to defer but didn't say for how long
    await _await_duration(user_id, state)
    return _ASK_DURATION_TEXT


async def _issue_all(user_id: str, state: dict) -> str:
    """Issue every order the open ask covered — one at a time, through the
    same guarded path the chat tool uses.

    The state this works from was written when the ask went out, so by now
    some of those orders may already have been invoiced (the seller did it
    in inFakt, or answered "tak" twice and both replies got this far).
    Orders already in the ledger are therefore dropped up front, and
    issue_invoice_for_order refuses any that slip through the gap between
    this check and its own claim.
    """
    from config.settings import get_settings
    from services import invoice_ledger
    from services.allegro_service import AllegroService
    from services.infakt_service import issue_invoice_for_order

    order_ids = state.get("order_ids", [])
    if order_ids:
        already = await invoice_ledger.already_handled(user_id, order_ids)
        order_ids = [oid for oid in order_ids if oid not in already]
    if not order_ids:
        await _resolve_state(user_id, state)
        return "Nie znalazłem już żadnych niewystawionych faktur do wystawienia — chyba ktoś mnie ubiegł. 🙂"

    allegro = AllegroService.get_instance(user_id)
    await allegro._load_tokens_from_redis()
    is_production = get_settings().is_production

    results = [await issue_invoice_for_order(allegro, order_id, is_production) for order_id in order_ids]

    await _resolve_state(user_id, state)
    return f"Wystawiam {_count_phrase(len(order_ids))}:\n\n" + "\n\n---\n\n".join(results)


# ── Reply classification (small dedicated LLM call, same shape as the
# orchestrator's own context classifier — see agents/orchestrator.py) ────────

_CLASSIFY_SYSTEM_TEMPLATE = """
Jesteś klasyfikatorem odpowiedzi sprzedawcy Allegro na automatyczne przypomnienie
asystenta o niewystawionych fakturach VAT.

KONTEKST: Asystent zapytał sprzedawcę, czy chce TERAZ wystawić {count} niewystawionych
faktur VAT dla już wysłanych zamówień.{extra}

Sklasyfikuj wiadomość sprzedawcy do JEDNEJ z poniższych akcji i odpowiedz DOKŁADNIE
jednym z poniższych formatów — nic więcej, żadnych wyjaśnień:

ISSUE
  — sprzedawca chce wystawić faktury teraz (np. "tak", "wystaw", "zrób to", "dawaj",
    "ok wystaw je", "proszę bardzo").
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

Odpowiedz TYLKO jednym z: ISSUE / SNOOZE:<liczba> / SNOOZE_UNSPECIFIED / DECLINE / UNRELATED.
""".strip()

# Second layer under _reminder_owns_reply: even when the reminder may claim
# the message, the assistant's last turn tells the classifier what the seller
# was most likely answering. Truncated because a last turn can be a full order
# listing, and only its opening (where the question would be) matters here.
_LAST_TURN_CHARS = 600
_LAST_TURN_TEMPLATE = (
    "OSTATNIA WIADOMOŚĆ ASYSTENTA w tej rozmowie — to na nią sprzedawca "
    "najprawdopodobniej odpowiada:\n\"\"\"{last}\"\"\"\n"
    "Jeśli wiadomość sprzedawcy odpowiada na TO pytanie, a nie na przypomnienie "
    "o fakturach (np. asystent zapytał o coś zupełnie innego, a sprzedawca "
    "odpowiedział „tak”), odpowiedz UNRELATED."
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
    count = len(state.get("order_ids", []))
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
        # without it a degraded-but-not-erroring model can sit inside one
        # call for up to the SDK's 600s default with no rotation.
        timeout=30.0,
    )
    try:
        resp = await _call_with_retry(
            client,
            settings.model_fast_pool(),
            "invoice_reminder/classify",
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
        logger.warning("Invoice reminder: classify LLM call failed: %s", exc)
        return "unrelated", 0

    return _parse_classification(raw)


def _parse_classification(raw: str) -> tuple[str, int]:
    if raw.startswith("ISSUE"):
        return "issue", 0
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
    r, user_id: str, *, status: str, order_ids: list[str],
    next_check_at: datetime, interval_minutes: int, reminder_count: int,
) -> None:
    payload = {
        "status": status,
        "order_ids": order_ids,
        "next_check_at": next_check_at.isoformat(),
        "interval_minutes": interval_minutes,
        "reminder_count": reminder_count,
    }
    await r.set(_STATE_KEY.format(user_id=user_id), json.dumps(payload), ex=_STATE_TTL)


async def _with_redis(fn) -> None:
    """Run `fn(r)` against a fresh Redis connection, closing it afterwards.
    Small helper for the one-off state writes triggered by a chat reply
    (as opposed to the cron pass, which already holds a connection)."""
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
            r, user_id, status=_STATUS_IDLE, order_ids=state.get("order_ids", []),
            next_check_at=now + timedelta(minutes=minutes),
            interval_minutes=minutes, reminder_count=0,
        )

    await _with_redis(_do)


async def _await_duration(user_id: str, state: dict) -> None:
    next_check_at = _parse_dt(state.get("next_check_at")) or datetime.now(_TZ)

    async def _do(r):
        await _save_state(
            r, user_id, status=_STATUS_AWAITING_DURATION,
            order_ids=state.get("order_ids", []),
            next_check_at=next_check_at,
            interval_minutes=state.get("interval_minutes", _DEFAULT_INTERVAL_MINUTES),
            reminder_count=state.get("reminder_count", 0),
        )

    await _with_redis(_do)


async def _resolve_state(user_id: str, state: dict) -> None:
    """Clear the outstanding ask after it's been acted on (issued), keeping
    the current cadence for the next check."""
    now = datetime.now(_TZ)
    interval = state.get("interval_minutes", _DEFAULT_INTERVAL_MINUTES)

    async def _do(r):
        await _save_state(
            r, user_id, status=_STATUS_IDLE, order_ids=[],
            next_check_at=now + timedelta(minutes=interval),
            interval_minutes=interval, reminder_count=0,
        )

    await _with_redis(_do)


async def _clear_state(user_id: str) -> None:
    async def _do(r):
        await r.delete(_STATE_KEY.format(user_id=user_id))

    await _with_redis(_do)
