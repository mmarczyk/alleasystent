"""Unit tests for services/message_reminder.py — the chat nudge that keeps
asking for as long as a buyer message stays unread.

Sibling of the invoice reminder, so the risky parts are the same: the
classifier's output parser must never invent an action, and an open reminder
must not claim a reply that answers a DIFFERENT open question.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


def _thread(tid="t1", login="kowalski", at="2026-09-02T06:34:00Z", unread=True):
    return {
        "id": tid,
        "read": not unread,
        "interlocutor": {"login": login},
        "lastMessageDateTime": at,
    }


class TestParseClassification:
    """Anything the parser can't read confidently must fall back to
    'unrelated' — that hands the message back to normal routing, which is the
    outcome that loses nothing."""

    def test_show(self):
        from services.message_reminder import _parse_classification
        assert _parse_classification("SHOW") == ("show", 0)

    def test_decline(self):
        from services.message_reminder import _parse_classification
        assert _parse_classification("DECLINE") == ("decline", 0)

    def test_snooze_with_minutes(self):
        from services.message_reminder import _parse_classification
        assert _parse_classification("SNOOZE:180") == ("snooze_duration", 180)

    def test_snooze_unspecified(self):
        from services.message_reminder import _parse_classification
        assert _parse_classification("SNOOZE_UNSPECIFIED") == ("snooze_unspecified", 0)

    def test_snooze_without_digits_falls_back(self):
        from services.message_reminder import _parse_classification
        assert _parse_classification("SNOOZE:") == ("snooze_unspecified", 0)
        assert _parse_classification("SNOOZE:jutro") == ("snooze_unspecified", 0)

    def test_garbage_defaults_to_unrelated(self):
        from services.message_reminder import _parse_classification
        assert _parse_classification("") == ("unrelated", 0)
        assert _parse_classification("MOŻE POKAŻ") == ("unrelated", 0)
        # Only a response that STARTS with SHOW counts — a mention mid-sentence
        # is the model narrating, not deciding.
        assert _parse_classification("I THINK YOU SHOULD SHOW THEM") == ("unrelated", 0)


class TestReminderOwnsReply:
    """A bare 'tak' answers the question the seller just read, not a nudge from
    hours ago — the same rule the invoice reminder needed after a stray 'tak'
    issued real VAT invoices."""

    def test_claims_when_last_turn_was_its_own_ask(self):
        from services.message_reminder import _reminder_owns_reply
        assert _reminder_owns_reply("tak", "💬 Masz 2 nieprzeczytane wiadomości od kupujących. Pokazać je?")

    def test_claims_when_last_turn_was_its_duration_follow_up(self):
        from services.message_reminder import _reminder_owns_reply, _ASK_DURATION_TEXT
        assert _reminder_owns_reply("2 godziny", _ASK_DURATION_TEXT)

    def test_claims_when_there_is_no_assistant_turn_to_answer(self):
        """The cross-thread case this module is built for."""
        from services.message_reminder import _reminder_owns_reply
        assert _reminder_owns_reply("tak", None)
        assert _reminder_owns_reply("tak", "   ")

    def test_does_not_claim_a_yes_meant_for_another_question(self):
        from services.message_reminder import _reminder_owns_reply
        assert not _reminder_owns_reply("tak", "Masz 3 nowe zamówienia. Pokazać szczegóły?")

    def test_claims_when_the_seller_names_messages_themselves(self):
        from services.message_reminder import _reminder_owns_reply
        assert _reminder_owns_reply("pokaż wiadomości", "Masz 3 nowe zamówienia. Pokazać szczegóły?")


class TestHandleReply:
    async def _handle(self, text, action, state=None, last_assistant=None):
        import services.message_reminder as mr

        state = state or {"status": "awaiting_response", "thread_ids": ["t1"], "interval_minutes": 120}
        with patch.object(mr, "get_pending_state", new=AsyncMock(return_value=state)), \
             patch.object(mr, "_classify_reply", new=AsyncMock(return_value=action)), \
             patch.object(mr, "_show_all", new=AsyncMock(return_value="LISTA")) as show, \
             patch.object(mr, "_set_snooze", new=AsyncMock()) as snooze, \
             patch.object(mr, "_await_duration", new=AsyncMock()) as await_dur, \
             patch.object(mr, "set_monitor_enabled", new=AsyncMock()) as set_enabled:
            result = await mr.handle_reply("user-1", text, last_assistant)
        return result, show, snooze, await_dur, set_enabled

    async def test_no_open_reminder_falls_through(self):
        import services.message_reminder as mr
        with patch.object(mr, "get_pending_state", new=AsyncMock(return_value=None)):
            assert await mr.handle_reply("user-1", "tak") is None

    async def test_show_lists_threads(self):
        result, show, *_ = await self._handle("tak", ("show", 0))
        assert result == "LISTA"
        show.assert_awaited_once()

    async def test_unrelated_falls_through_to_normal_routing(self):
        result, show, *_ = await self._handle("ile mam zamówień?", ("unrelated", 0))
        assert result is None
        show.assert_not_awaited()

    async def test_decline_turns_the_reminder_off(self):
        result, _, _, _, set_enabled = await self._handle("przestań pytać", ("decline", 0))
        assert set_enabled.await_args.args == ("user-1", False)
        assert "wyłączyłem" in result

    async def test_snooze_confirms_the_duration(self):
        result, _, snooze, *_ = await self._handle("za 3 godziny", ("snooze_duration", 180))
        assert snooze.await_args.args[2] == 180
        assert "3 godziny" in result

    async def test_snooze_is_clamped_to_sane_bounds(self):
        import services.message_reminder as mr
        _, _, snooze, *_ = await self._handle("za chwilę", ("snooze_duration", 1))
        assert snooze.await_args.args[2] == mr._MIN_SNOOZE_MINUTES
        _, _, snooze, *_ = await self._handle("za rok", ("snooze_duration", 999_999))
        assert snooze.await_args.args[2] == mr._MAX_SNOOZE_MINUTES

    async def test_vague_deferral_asks_how_long(self):
        import services.message_reminder as mr
        result, _, _, await_dur, _ = await self._handle("później", ("snooze_unspecified", 0))
        assert result == mr._ASK_DURATION_TEXT
        await_dur.assert_awaited_once()

    async def test_reply_for_another_question_never_reaches_the_classifier(self):
        import services.message_reminder as mr
        state = {"status": "awaiting_response", "thread_ids": ["t1"]}
        with patch.object(mr, "get_pending_state", new=AsyncMock(return_value=state)), \
             patch.object(mr, "_classify_reply", new=AsyncMock()) as classify:
            result = await mr.handle_reply("user-1", "tak", "Masz 3 nowe zamówienia. Pokazać szczegóły?")
        assert result is None
        classify.assert_not_awaited()


class TestShowAll:
    async def test_lists_what_is_unread_now_not_what_was_asked_about(self):
        """The seller may have read some threads since the ask; replaying the
        stored IDs would show messages that are no longer unread."""
        import services.message_reminder as mr

        allegro = MagicMock()
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_message_threads = AsyncMock(return_value=[
            _thread("t1", "kowalski", "2026-09-02T06:34:00Z"),
            _thread("t2", "nowak", "2026-09-01T17:12:00Z", unread=False),
        ])
        with patch("services.allegro_service.AllegroService") as svc, \
             patch.object(mr, "_resolve_state", new=AsyncMock()):
            svc.get_instance.return_value = allegro
            out = await mr._show_all("user-1", {"thread_ids": ["t1", "t2"], "interval_minutes": 120})
        assert "kowalski" in out and "nowak" not in out
        assert "1 nieprzeczytaną wiadomość" in out
        assert "`t1`" in out  # the ID stays, send_message_to_buyer reads it back

    async def test_everything_read_since_the_ask(self):
        import services.message_reminder as mr

        allegro = MagicMock()
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_message_threads = AsyncMock(return_value=[_thread(unread=False)])
        with patch("services.allegro_service.AllegroService") as svc, \
             patch.object(mr, "_resolve_state", new=AsyncMock()):
            svc.get_instance.return_value = allegro
            out = await mr._show_all("user-1", {"thread_ids": ["t1"]})
        assert "Nie masz już żadnych nieprzeczytanych" in out

    async def test_api_error_does_not_raise_into_the_chat(self):
        import services.message_reminder as mr
        from services.allegro_service import AllegroAPIError

        allegro = MagicMock()
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_message_threads = AsyncMock(side_effect=AllegroAPIError(502, "boom"))
        with patch("services.allegro_service.AllegroService") as svc, \
             patch.object(mr, "_resolve_state", new=AsyncMock()) as resolve:
            svc.get_instance.return_value = allegro
            out = await mr._show_all("user-1", {"thread_ids": ["t1"]})
        assert "Nie udało mi się" in out
        # The ask stays open, so the next pass nags again rather than losing it.
        resolve.assert_not_awaited()


class TestFormatting:
    def test_unread_phrase_polish_plurals(self):
        from services.message_reminder import _unread_phrase
        assert _unread_phrase(1) == "1 nieprzeczytaną wiadomość"
        assert _unread_phrase(3) == "3 nieprzeczytanych wiadomości"

    def test_thread_line_renders_warsaw_local_time(self):
        from services.message_reminder import _format_threads
        line = _format_threads([_thread("t1", "kowalski", "2026-09-02T06:34:00Z")])
        assert "kowalski" in line and "`t1`" in line
        assert "02.09.2026, 08:34" in line  # 06:34 UTC → 08:34 Warsaw (CEST)

    def test_missing_timestamp_does_not_crash(self):
        from services.message_reminder import _format_threads
        assert "—" in _format_threads([{"id": "t1", "read": False}])


class TestFetchUnread:
    async def test_filters_read_threads_and_sorts_newest_first(self):
        import services.message_reminder as mr

        allegro = MagicMock()
        allegro.get_message_threads = AsyncMock(return_value=[
            _thread("old", "a", "2026-09-01T10:00:00Z"),
            _thread("read", "b", "2026-09-02T10:00:00Z", unread=False),
            _thread("new", "c", "2026-09-02T09:00:00Z"),
        ])
        out = await mr._fetch_unread(allegro)
        assert [t["id"] for t in out] == ["new", "old"]

    async def test_uses_the_shared_unread_predicate_not_a_local_field_name(self):
        """The monitor lost months of notifications to a hand-written field
        name; this path must go through services.allegro_service."""
        import inspect
        import services.message_reminder as mr

        src = inspect.getsource(mr)
        assert "is_thread_unread" in src
        assert "hasUnreadMessages" not in src
        assert "lastMessageCreatedAt" not in src


class TestEnabledFlag:
    async def test_uses_its_own_kind_distinct_from_the_monitor(self):
        """Sharing the message MONITOR's flag would make one toggle silently
        switch the other."""
        import services.message_reminder as mr
        import services.message_monitor as mm

        assert mr._MONITOR_KIND != mm._MONITOR_KIND
        with patch("services.monitor_state.is_monitor_enabled", new=AsyncMock(return_value=True)) as chk:
            assert await mr.is_monitor_enabled("user-1") is True
        assert chk.await_args.args == ("message_reminder", "user-1")

    async def test_disabling_clears_the_open_ask(self):
        """Otherwise a re-enable would resume mid-conversation, answering a
        question the seller never saw."""
        import services.message_reminder as mr
        with patch("services.monitor_state.set_monitor_enabled", new=AsyncMock()), \
             patch.object(mr, "_clear_state", new=AsyncMock()) as clear:
            await mr.set_monitor_enabled("user-1", False)
        clear.assert_awaited_once()

    async def test_enabling_keeps_state_untouched(self):
        import services.message_reminder as mr
        with patch("services.monitor_state.set_monitor_enabled", new=AsyncMock()), \
             patch.object(mr, "_clear_state", new=AsyncMock()) as clear:
            await mr.set_monitor_enabled("user-1", True)
        clear.assert_not_awaited()


class TestBusinessHours:
    async def test_quiet_outside_working_hours(self):
        """No 3am nudges — same gate as the invoice reminder."""
        import services.message_reminder as mr
        from datetime import datetime

        with patch.object(mr, "_poll_all_users", new=AsyncMock()) as poll, \
             patch("config.settings.get_settings") as settings:
            settings.return_value.redis_url = "redis://localhost:6379"
            with patch.object(mr, "datetime") as dt:
                dt.now.return_value = datetime(2026, 9, 2, 3, 0, tzinfo=mr._TZ)
                await mr.run_once()
        poll.assert_not_awaited()

    async def test_runs_during_working_hours(self):
        import services.message_reminder as mr
        from datetime import datetime

        with patch.object(mr, "_poll_all_users", new=AsyncMock()) as poll, \
             patch("config.settings.get_settings") as settings:
            settings.return_value.redis_url = "redis://localhost:6379"
            with patch.object(mr, "datetime") as dt:
                dt.now.return_value = datetime(2026, 9, 2, 10, 0, tzinfo=mr._TZ)
                await mr.run_once()
        poll.assert_awaited_once()

    async def test_no_redis_is_a_no_op(self):
        import services.message_reminder as mr
        with patch.object(mr, "_poll_all_users", new=AsyncMock()) as poll:
            await mr.run_once()
        poll.assert_not_awaited()


class TestPollUser:
    """The cron half: when it asks, when it re-asks, and when it goes quiet."""

    @staticmethod
    def _redis(state=None, has_tokens=True):
        r = MagicMock()
        r.exists = AsyncMock(return_value=1 if has_tokens else 0)
        r.get = AsyncMock(return_value=state)
        r.set = AsyncMock()
        return r

    async def _poll(self, threads, state=None, now_hour=10):
        import json
        from datetime import datetime
        import services.message_reminder as mr

        r = self._redis(json.dumps(state) if state else None)
        allegro = MagicMock()
        allegro._tokens = {"access_token": "x"}
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_message_threads = AsyncMock(return_value=threads)
        now = datetime(2026, 9, 2, now_hour, 0, tzinfo=mr._TZ)

        with patch("services.allegro_service.AllegroService") as svc, \
             patch.object(mr, "_ask", new=AsyncMock()) as ask:
            svc.get_instance.return_value = allegro
            await mr._poll_user(r, "user-1", now)
        saved = json.loads(r.set.await_args.args[1]) if r.set.await_args else None
        return ask, saved

    async def test_first_unread_asks_once(self):
        ask, saved = await self._poll([_thread()])
        ask.assert_awaited_once()
        assert ask.await_args.kwargs["again"] is False
        assert saved["status"] == "awaiting_response"
        assert saved["thread_ids"] == ["t1"]
        assert saved["reminder_count"] == 1

    async def test_unanswered_ask_nags_again(self):
        ask, saved = await self._poll(
            [_thread()],
            state={"status": "awaiting_response", "thread_ids": ["t1"],
                   "interval_minutes": 120, "reminder_count": 1,
                   "next_check_at": "2026-09-02T08:00:00+02:00"},
        )
        assert ask.await_args.kwargs["again"] is True
        assert saved["reminder_count"] == 2

    async def test_everything_read_goes_quiet(self):
        ask, saved = await self._poll([_thread(unread=False)])
        ask.assert_not_awaited()
        assert saved["status"] == "idle"
        assert saved["thread_ids"] == []

    async def test_not_due_yet_does_nothing(self):
        """The seller's own snooze must actually hold."""
        ask, saved = await self._poll(
            [_thread()],
            state={"status": "idle", "thread_ids": [], "interval_minutes": 240,
                   "next_check_at": "2026-09-02T23:00:00+02:00"},
        )
        ask.assert_not_awaited()
        assert saved is None

    async def test_snoozed_interval_is_carried_forward(self):
        """A seller who said "za 4 godziny" keeps that cadence, not the default."""
        _, saved = await self._poll(
            [_thread()],
            state={"status": "idle", "thread_ids": [], "interval_minutes": 240,
                   "next_check_at": "2026-09-02T08:00:00+02:00"},
        )
        assert saved["interval_minutes"] == 240

    async def test_user_without_allegro_tokens_is_skipped(self):
        from datetime import datetime
        import services.message_reminder as mr

        r = self._redis(has_tokens=False)
        with patch.object(mr, "_ask", new=AsyncMock()) as ask:
            await mr._poll_user(r, "user-1", datetime(2026, 9, 2, 10, 0, tzinfo=mr._TZ))
        ask.assert_not_awaited()


class TestDeliveryChannel:
    async def test_delivered_as_chat_only_never_as_a_push(self):
        """The reminder is the assistant writing in the chat. The message
        MONITOR is what pushes; if this pushed too, one unread message would
        alert twice."""
        import services.message_reminder as mr

        with patch("services.push_service.store_pending_chat", new=AsyncMock()) as chat, \
             patch("services.push_service.send_push", new=AsyncMock()) as push, \
             patch("services.push_service.add_notification", new=AsyncMock()) as inbox:
            await mr._ask("user-1", [_thread()], again=False, awaiting_duration=False)
        chat.assert_awaited_once()
        push.assert_not_awaited()
        inbox.assert_not_awaited()
        assert chat.await_args.kwargs["dedupe_tag"] == mr._MONITOR_KIND

    async def test_ask_names_the_buyers(self):
        import services.message_reminder as mr

        with patch("services.push_service.store_pending_chat", new=AsyncMock()) as chat:
            await mr._ask("user-1", [_thread(login="kowalski")], again=False, awaiting_duration=False)
        text = chat.await_args.args[1]
        assert "1 nieprzeczytaną wiadomość" in text and "kowalski" in text
