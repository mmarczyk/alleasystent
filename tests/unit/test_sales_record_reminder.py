"""Unit tests for services/sales_record_reminder.py — the monthly nag about the
ewidencja sprzedaży bezrachunkowej for the previous month.

The two things worth testing hard are the ones that make this reminder
different from its siblings: the cadence is the calendar (twice a day over the
1st-3rd, four times a day from the 4th, 8:00-20:00) rather than an interval,
and NOTHING can confirm the work is done except the seller saying so — which
is why "już wystawiłem" has to be recognised without an LLM call, and why a
month that rolls over unconfirmed must not simply vanish.
"""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


def _now(day: int, hour: int, minute: int = 5, month: int = 9, year: int = 2026):
    import services.sales_record_reminder as sr
    return datetime(year, month, day, hour, minute, tzinfo=sr._TZ)


class TestParseClassification:
    """Anything the parser can't read confidently must fall back to
    'unrelated' — that hands the message back to normal routing, which is the
    outcome that loses nothing."""

    def test_done(self):
        from services.sales_record_reminder import _parse_classification
        assert _parse_classification("DONE") == ("done", 0)

    def test_decline(self):
        from services.sales_record_reminder import _parse_classification
        assert _parse_classification("DECLINE") == ("decline", 0)

    def test_snooze_with_minutes(self):
        from services.sales_record_reminder import _parse_classification
        assert _parse_classification("SNOOZE:180") == ("snooze_duration", 180)

    def test_snooze_unspecified(self):
        from services.sales_record_reminder import _parse_classification
        assert _parse_classification("SNOOZE_UNSPECIFIED") == ("snooze_unspecified", 0)

    def test_snooze_without_digits_falls_back(self):
        from services.sales_record_reminder import _parse_classification
        assert _parse_classification("SNOOZE:jutro") == ("snooze_unspecified", 0)

    def test_garbage_defaults_to_unrelated(self):
        from services.sales_record_reminder import _parse_classification
        assert _parse_classification("") == ("unrelated", 0)
        assert _parse_classification("CHYBA DONE") == ("unrelated", 0)


class TestLooksAlreadyDone:
    """The seller's word is the only thing that can stop this reminder, so the
    common phrasings must land even when the classifier is unavailable."""

    @pytest.mark.parametrize("text", [
        "już wystawiłem",
        "wystawiłem już",
        "już wysłane",
        "już gotowe",
        "gotowe",
        "zrobione",
        "już to wysłałem księgowej",
        "już wystawione, dzięki",
    ])
    def test_confirmations(self, text):
        from services.sales_record_reminder import _looks_already_done
        assert _looks_already_done(text)

    @pytest.mark.parametrize("text", [
        "jeszcze nie wystawiłem",
        "nie pamiętam, czy już wystawiłem",
        "czy już gotowe?",
        "zaraz to zrobię",
        "później",
        "tak",
        "przypomnij jutro",
    ])
    def test_not_confirmations(self, text):
        """Everything here goes to the classifier instead — nagging someone who
        has finished costs a message, stopping for someone who hasn't costs
        them the deadline."""
        from services.sales_record_reminder import _looks_already_done
        assert not _looks_already_done(text)


class TestReminderOwnsReply:
    def test_claims_when_last_turn_was_its_own_ask(self):
        import services.sales_record_reminder as sr
        ask = sr._build_ask_text("2026-08", [], _now(1, 8), again=False, awaiting_duration=False)
        assert sr._reminder_owns_reply("już wystawiłem", ask)

    def test_claims_when_last_turn_was_its_duration_follow_up(self):
        import services.sales_record_reminder as sr
        assert sr._reminder_owns_reply("2 godziny", sr._ASK_DURATION_TEXT)

    def test_claims_when_there_is_no_assistant_turn_to_answer(self):
        import services.sales_record_reminder as sr
        assert sr._reminder_owns_reply("już wystawiłem", None)

    def test_does_not_claim_a_yes_meant_for_another_question(self):
        import services.sales_record_reminder as sr
        assert not sr._reminder_owns_reply("tak", "Masz 3 nowe zamówienia. Pokazać szczegóły?")

    def test_claims_when_the_seller_names_the_ewidencja(self):
        import services.sales_record_reminder as sr
        assert sr._reminder_owns_reply(
            "ewidencja już wysłana", "Masz 3 nowe zamówienia. Pokazać szczegóły?",
        )

    def test_own_ask_re_does_not_match_the_other_reminders(self):
        """A generic duration follow-up would snooze whichever reminder was
        consulted first — see services/invoice_reminder.py._OWN_ASK_RE."""
        import services.sales_record_reminder as sr
        from services.invoice_reminder import _ASK_DURATION_TEXT as INVOICE_ASK
        from services.message_reminder import _ASK_DURATION_TEXT as MESSAGE_ASK

        assert not sr._OWN_ASK_RE.search(INVOICE_ASK)
        assert not sr._OWN_ASK_RE.search(MESSAGE_ASK)
        assert sr._OWN_ASK_RE.search(sr._ASK_DURATION_TEXT)


class TestCalendar:
    def test_target_period_is_always_the_month_that_just_ended(self):
        import services.sales_record_reminder as sr
        assert sr._target_period(_now(1, 8)) == "2026-08"
        assert sr._target_period(_now(28, 8)) == "2026-08"
        assert sr._target_period(_now(3, 8, month=1)) == "2025-12"

    def test_cadence_tightens_after_the_third(self):
        import services.sales_record_reminder as sr
        assert sr._slot_hours(1) == (8, 20)
        assert sr._slot_hours(3) == (8, 20)
        assert sr._slot_hours(4) == (8, 12, 16, 20)
        assert sr._slot_hours(28) == (8, 12, 16, 20)

    def test_no_slot_before_eight(self):
        import services.sales_record_reminder as sr
        assert sr._due_slot(_now(1, 7, 59)) is None

    def test_slot_is_the_most_recent_one_passed(self):
        """A missed pass delivers late rather than skipping the slot."""
        import services.sales_record_reminder as sr
        assert sr._due_slot(_now(1, 11)).hour == 8      # 1st: 8:00 and 20:00 only
        assert sr._due_slot(_now(1, 20, 3)).hour == 20
        assert sr._due_slot(_now(7, 13)).hour == 12     # 7th: every four hours
        assert sr._due_slot(_now(7, 19, 59)).hour == 16

    def test_days_to_deadline(self):
        import services.sales_record_reminder as sr
        assert sr._days_to_deadline(_now(1, 8)) == 4
        assert sr._days_to_deadline(_now(5, 8)) == 0
        assert sr._days_to_deadline(_now(9, 8)) == -4

    def test_deadline_sentence_flips_once_the_fifth_has_passed(self):
        import services.sales_record_reminder as sr
        assert "zostały 4 dni" in sr._deadline_sentence(_now(1, 8))
        assert "został 1 dzień" in sr._deadline_sentence(_now(4, 8))
        assert "czyli dzisiaj" in sr._deadline_sentence(_now(5, 8))
        assert "minął 4 dni temu" in sr._deadline_sentence(_now(9, 8))


class TestAskText:
    def test_first_ask_names_the_month_and_the_deadline(self):
        import services.sales_record_reminder as sr
        text = sr._build_ask_text("2026-08", [], _now(1, 8), again=False, awaiting_duration=False)
        assert "sierpień 2026" in text
        assert "5 września" in text
        assert "już wystawiłem" in text

    def test_overdue_months_are_named_in_every_later_ask(self):
        import services.sales_record_reminder as sr
        text = sr._build_ask_text("2026-08", ["2026-07"], _now(9, 12), again=True, awaiting_duration=False)
        assert "lipiec 2026" in text
        assert text.startswith("⚠️")  # past the deadline


class TestBusinessHours:
    async def test_quiet_outside_the_window(self):
        import services.sales_record_reminder as sr
        with patch.object(sr, "_poll_all_users", new=AsyncMock()) as poll, \
             patch("config.settings.get_settings") as settings:
            settings.return_value.redis_url = "redis://localhost:6379"
            with patch.object(sr, "datetime") as dt:
                dt.now.return_value = _now(2, 3)
                await sr.run_once()
        poll.assert_not_awaited()

    async def test_the_eight_pm_slot_survives_a_late_pass(self):
        """The gate is inclusive of 20:xx or the last ask of the day is lost."""
        import services.sales_record_reminder as sr
        with patch.object(sr, "_poll_all_users", new=AsyncMock()) as poll, \
             patch("config.settings.get_settings") as settings:
            settings.return_value.redis_url = "redis://localhost:6379"
            with patch.object(sr, "datetime") as dt:
                dt.now.return_value = _now(2, 20, 3)
                await sr.run_once()
        poll.assert_awaited_once()

    async def test_no_redis_is_a_no_op(self):
        import services.sales_record_reminder as sr
        with patch.object(sr, "_poll_all_users", new=AsyncMock()) as poll:
            await sr.run_once()
        poll.assert_not_awaited()


class TestPollUser:
    """The cron half: when it asks, when it stays quiet, and what it remembers."""

    @staticmethod
    def _redis(state=None):
        r = MagicMock()
        r.get = AsyncMock(return_value=json.dumps(state) if state else None)
        r.set = AsyncMock()
        r.exists = AsyncMock(return_value=1)
        return r

    async def _poll(self, now, state=None):
        import services.sales_record_reminder as sr

        r = self._redis(state)
        with patch.object(sr, "_ask", new=AsyncMock()) as ask:
            await sr._poll_user(r, "user-1", now)
        saved = json.loads(r.set.await_args.args[1]) if r.set.await_args else None
        return ask, saved

    async def test_first_day_of_the_month_asks_about_the_previous_one(self):
        ask, saved = await self._poll(_now(1, 8))
        ask.assert_awaited_once()
        assert ask.await_args.args[1] == "2026-08"
        assert ask.await_args.kwargs["again"] is False
        assert saved["status"] == "awaiting_response"
        assert saved["period"] == "2026-08"
        assert saved["reminder_count"] == 1

    async def test_never_asks_twice_inside_the_same_slot(self):
        _, saved = await self._poll(
            _now(1, 9),
            state={"status": "awaiting_response", "period": "2026-08",
                   "last_ask_at": _now(1, 8, 1).isoformat(), "reminder_count": 1},
        )
        assert saved is None

    async def test_the_next_slot_of_the_day_asks_again(self):
        ask, saved = await self._poll(
            _now(1, 20),
            state={"status": "awaiting_response", "period": "2026-08",
                   "last_ask_at": _now(1, 8, 1).isoformat(), "reminder_count": 1},
        )
        assert ask.await_args.kwargs["again"] is True
        assert saved["reminder_count"] == 2

    async def test_four_slots_a_day_from_the_fourth(self):
        """The 12:00 ask exists only in the tighter cadence — on the 1st-3rd
        the same state would have to wait for 20:00."""
        ask, _ = await self._poll(
            _now(4, 12),
            state={"status": "awaiting_response", "period": "2026-08",
                   "last_ask_at": _now(4, 8, 1).isoformat(), "reminder_count": 2},
        )
        ask.assert_awaited_once()

        ask, saved = await self._poll(
            _now(3, 12),
            state={"status": "awaiting_response", "period": "2026-08",
                   "last_ask_at": _now(3, 8, 1).isoformat(), "reminder_count": 2},
        )
        ask.assert_not_awaited()
        assert saved is None

    async def test_a_confirmed_month_is_never_asked_about_again(self):
        ask, saved = await self._poll(
            _now(9, 12),
            state={"status": "idle", "period": "2026-08", "done_periods": ["2026-08"],
                   "last_ask_at": _now(2, 8).isoformat(), "reminder_count": 3},
        )
        ask.assert_not_awaited()
        assert saved is None

    async def test_snooze_holds_the_asks_off(self):
        ask, saved = await self._poll(
            _now(4, 12),
            state={"status": "idle", "period": "2026-08",
                   "snooze_until": _now(4, 18).isoformat(),
                   "last_ask_at": _now(4, 8).isoformat(), "reminder_count": 2},
        )
        ask.assert_not_awaited()
        assert saved is None

    async def test_an_expired_snooze_reopens_the_ask(self):
        """Otherwise the state stays idle and the seller's "już wystawiłem"
        would not be claimed by this reminder at all."""
        ask, saved = await self._poll(
            _now(4, 16),
            state={"status": "idle", "period": "2026-08",
                   "snooze_until": _now(4, 15).isoformat(),
                   "last_ask_at": _now(4, 8).isoformat(), "reminder_count": 2},
        )
        ask.assert_awaited_once()
        assert saved["status"] == "awaiting_response"
        assert saved["snooze_until"] is None

    async def test_a_new_month_starts_over_and_carries_the_unconfirmed_one(self):
        ask, saved = await self._poll(
            _now(1, 8, month=10),
            state={"status": "awaiting_response", "period": "2026-08",
                   "last_ask_at": _now(30, 20).isoformat(), "reminder_count": 12},
        )
        assert ask.await_args.args[1] == "2026-09"
        assert ask.await_args.args[2] == ["2026-08"]   # still owed, still named
        assert ask.await_args.kwargs["again"] is False
        assert saved["period"] == "2026-09"
        assert saved["overdue"] == ["2026-08"]
        assert saved["reminder_count"] == 1

    async def test_a_month_confirmed_before_the_rollover_is_not_carried(self):
        ask, saved = await self._poll(
            _now(1, 8, month=10),
            state={"status": "idle", "period": "2026-08", "done_periods": ["2026-08"],
                   "last_ask_at": _now(2, 8).isoformat(), "reminder_count": 3},
        )
        assert ask.await_args.args[2] == []
        assert saved["overdue"] == []
        assert saved["done_periods"] == ["2026-08"]

    async def test_runs_without_any_allegro_call_or_token(self):
        """The whole point of this reminder: the obligation is the seller's
        whether or not their Allegro connection is alive, and nothing outside
        Redis knows anything about it."""
        import inspect
        import services.sales_record_reminder as sr

        src = inspect.getsource(sr)
        assert "AllegroService" not in src
        assert "allegro:tokens" not in src


class TestHandleReply:
    async def _handle(self, text, action=("unrelated", 0), state=None, last_assistant=None):
        import services.sales_record_reminder as sr

        state = state or {"status": "awaiting_response", "period": "2026-08", "overdue": []}
        with patch.object(sr, "get_pending_state", new=AsyncMock(return_value=state)), \
             patch.object(sr, "_classify_reply", new=AsyncMock(return_value=action)) as classify, \
             patch.object(sr, "_mark_done", new=AsyncMock()) as mark_done, \
             patch.object(sr, "_set_snooze", new=AsyncMock()) as snooze, \
             patch.object(sr, "_await_duration", new=AsyncMock()) as await_dur, \
             patch.object(sr, "set_monitor_enabled", new=AsyncMock()) as set_enabled:
            result = await sr.handle_reply("user-1", text, last_assistant)
        return result, classify, mark_done, snooze, await_dur, set_enabled

    async def test_no_open_reminder_falls_through(self):
        import services.sales_record_reminder as sr
        with patch.object(sr, "get_pending_state", new=AsyncMock(return_value=None)):
            assert await sr.handle_reply("user-1", "już wystawiłem") is None

    async def test_a_plain_confirmation_never_needs_the_classifier(self):
        result, classify, mark_done, *_ = await self._handle("już wystawiłem")
        classify.assert_not_awaited()
        assert mark_done.await_args.args[2] == ["2026-08"]
        assert "sierpień 2026" in result

    async def test_a_confirmation_settles_the_overdue_months_it_was_told_about(self):
        state = {"status": "awaiting_response", "period": "2026-08", "overdue": ["2026-07"]}
        result, _, mark_done, *_ = await self._handle("już wysłane", state=state)
        assert mark_done.await_args.args[2] == ["2026-08", "2026-07"]
        assert "lipiec 2026" in result

    async def test_classifier_done_marks_the_period_too(self):
        _, _, mark_done, *_ = await self._handle("księgowa to ogarnęła", ("done", 0))
        mark_done.assert_awaited_once()

    async def test_unrelated_falls_through_to_normal_routing(self):
        result, _, mark_done, *_ = await self._handle("ile mam zamówień?", ("unrelated", 0))
        assert result is None
        mark_done.assert_not_awaited()

    async def test_decline_turns_the_reminder_off(self):
        result, _, _, _, _, set_enabled = await self._handle("wyłącz to", ("decline", 0))
        assert set_enabled.await_args.args == ("user-1", False)
        assert "wyłączyłem" in result

    async def test_snooze_confirms_the_duration(self):
        result, _, _, snooze, *_ = await self._handle("za 3 godziny", ("snooze_duration", 180))
        assert snooze.await_args.args[2] == 180
        assert "3 godziny" in result

    async def test_snooze_is_clamped_to_sane_bounds(self):
        import services.sales_record_reminder as sr
        _, _, _, snooze, *_ = await self._handle("za chwilę", ("snooze_duration", 1))
        assert snooze.await_args.args[2] == sr._MIN_SNOOZE_MINUTES
        _, _, _, snooze, *_ = await self._handle("za rok", ("snooze_duration", 999_999))
        assert snooze.await_args.args[2] == sr._MAX_SNOOZE_MINUTES

    async def test_vague_deferral_asks_how_long(self):
        import services.sales_record_reminder as sr
        result, _, _, _, await_dur, _ = await self._handle("później", ("snooze_unspecified", 0))
        assert result == sr._ASK_DURATION_TEXT
        await_dur.assert_awaited_once()

    async def test_reply_for_another_question_never_reaches_the_classifier(self):
        result, classify, mark_done, *_ = await self._handle(
            "tak", last_assistant="Masz 3 nowe zamówienia. Pokazać szczegóły?",
        )
        assert result is None
        classify.assert_not_awaited()
        mark_done.assert_not_awaited()


class TestMarkDone:
    async def test_only_the_confirmed_months_are_remembered(self):
        import services.sales_record_reminder as sr

        r = MagicMock()
        r.set = AsyncMock()

        async def _with_redis(fn):
            await fn(r)

        with patch.object(sr, "_with_redis", new=_with_redis):
            await sr._mark_done(
                "user-1",
                {"period": "2026-08", "overdue": ["2026-07"], "done_periods": ["2026-06"]},
                ["2026-08"],
            )
        saved = json.loads(r.set.await_args.args[1])
        assert saved["done_periods"] == ["2026-06", "2026-08"]
        assert saved["overdue"] == ["2026-07"]   # not confirmed, so still owed
        assert saved["status"] == "idle"


class TestRefreshPendingMessage:
    """A queued ask states how many days are left until the 5th, and can sit in
    the queue for a day — so it is rebuilt from the clock at delivery."""

    async def test_rebuilds_the_day_count(self):
        import services.sales_record_reminder as sr

        state = {"status": "awaiting_response", "period": "2026-08", "overdue": [],
                 "reminder_count": 2}
        with patch.object(sr, "get_pending_state", new=AsyncMock(return_value=state)), \
             patch.object(sr, "datetime") as dt:
            dt.now.return_value = _now(9, 12)
            out = await sr.refresh_pending_message("user-1", "stara treść z 3. dnia")
        assert "minął 4 dni temu" in out

    async def test_dropped_when_the_ask_is_no_longer_open(self):
        import services.sales_record_reminder as sr
        with patch.object(sr, "get_pending_state", new=AsyncMock(return_value=None)):
            assert await sr.refresh_pending_message("user-1", "stara treść") is None


class TestEnabledFlag:
    async def test_uses_its_own_kind(self):
        import services.sales_record_reminder as sr
        from services import invoice_reminder, message_reminder

        assert sr._MONITOR_KIND not in (invoice_reminder._MONITOR_KIND, message_reminder._MONITOR_KIND)
        with patch("services.monitor_state.is_monitor_enabled", new=AsyncMock(return_value=True)) as chk:
            assert await sr.is_monitor_enabled("user-1") is True
        assert chk.await_args.args == ("sales_record_reminder", "user-1")

    async def test_disabling_clears_the_open_ask(self):
        import services.sales_record_reminder as sr
        with patch("services.monitor_state.set_monitor_enabled", new=AsyncMock()), \
             patch.object(sr, "_clear_state", new=AsyncMock()) as clear:
            await sr.set_monitor_enabled("user-1", False)
        clear.assert_awaited_once()


class TestDeliveryChannel:
    async def test_delivered_as_chat_only_never_as_a_push(self):
        import services.sales_record_reminder as sr

        with patch("services.push_service.store_pending_chat", new=AsyncMock()) as chat, \
             patch("services.push_service.send_push", new=AsyncMock()) as push, \
             patch("services.push_service.add_notification", new=AsyncMock()) as inbox:
            await sr._ask("user-1", "2026-08", [], _now(1, 8), again=False, awaiting_duration=False)
        chat.assert_awaited_once()
        assert chat.await_args.kwargs["dedupe_tag"] == sr._MONITOR_KIND
        push.assert_not_awaited()
        inbox.assert_not_awaited()
