"""Unit tests for services/invoice_reminder.py."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


class TestParseClassification:
    """The reply classifier's output parser — safety-critical: anything it
    can't confidently parse must fall back to "unrelated", never "issue"."""

    def test_issue(self):
        from services.invoice_reminder import _parse_classification
        assert _parse_classification("ISSUE") == ("issue", 0)

    def test_decline(self):
        from services.invoice_reminder import _parse_classification
        assert _parse_classification("DECLINE") == ("decline", 0)

    def test_snooze_unspecified(self):
        from services.invoice_reminder import _parse_classification
        assert _parse_classification("SNOOZE_UNSPECIFIED") == ("snooze_unspecified", 0)

    def test_snooze_with_minutes(self):
        from services.invoice_reminder import _parse_classification
        assert _parse_classification("SNOOZE:180") == ("snooze_duration", 180)

    def test_snooze_missing_digits_falls_back_to_unspecified(self):
        from services.invoice_reminder import _parse_classification
        assert _parse_classification("SNOOZE:") == ("snooze_unspecified", 0)
        assert _parse_classification("SNOOZE:abc") == ("snooze_unspecified", 0)

    def test_unrelated(self):
        from services.invoice_reminder import _parse_classification
        assert _parse_classification("UNRELATED") == ("unrelated", 0)

    def test_garbage_input_defaults_to_unrelated_never_issue(self):
        from services.invoice_reminder import _parse_classification
        assert _parse_classification("") == ("unrelated", 0)
        assert _parse_classification("SOMETHING WEIRD THE MODEL SAID") == ("unrelated", 0)
        # Only a response that actually STARTS with ISSUE counts — text merely
        # mentioning it mid-sentence must never be treated as a real issue command.
        assert _parse_classification("I THINK MAYBE ISSUE THEM") == ("unrelated", 0)


class TestFormatDuration:
    def test_minutes(self):
        from services.invoice_reminder import _format_duration
        assert _format_duration(1) == "1 minutę"
        assert _format_duration(20) == "20 minut"

    def test_hours(self):
        from services.invoice_reminder import _format_duration
        assert _format_duration(60) == "1 godzinę"
        assert _format_duration(120) == "2 godziny"
        assert _format_duration(180) == "3 godziny"
        assert _format_duration(300) == "5 godzin"

    def test_days(self):
        from services.invoice_reminder import _format_duration
        assert _format_duration(60 * 24) == "1 dzień"
        assert _format_duration(60 * 24 * 3) == "3 dni"


class TestPhrases:
    def test_pending_invoice_phrase_singular(self):
        from services.invoice_reminder import _pending_invoice_phrase
        assert _pending_invoice_phrase(1) == "1 niewystawioną fakturę"

    def test_pending_invoice_phrase_plural(self):
        from services.invoice_reminder import _pending_invoice_phrase
        assert _pending_invoice_phrase(3) == "3 niewystawionych faktur"

    def test_count_phrase(self):
        from services.invoice_reminder import _count_phrase
        assert _count_phrase(1) == "1 fakturę"
        assert _count_phrase(5) == "5 faktur"

    def test_format_order_ids_truncates(self):
        from services.invoice_reminder import _format_order_ids
        ids = [f"order-{i}" for i in range(12)]
        result = _format_order_ids(ids)
        assert "order-0" in result
        assert "order-9" in result
        assert "order-10" not in result
        assert "2 więcej" in result


class TestMonitorEnabled:
    @pytest.mark.asyncio
    async def test_is_monitor_enabled_delegates_to_monitor_state(self, monkeypatch):
        from services import invoice_reminder

        mock_is_enabled = AsyncMock(return_value=True)
        with patch("services.monitor_state.is_monitor_enabled", mock_is_enabled):
            result = await invoice_reminder.is_monitor_enabled("user1")
        assert result is True
        mock_is_enabled.assert_awaited_once_with("invoice_reminder", "user1")

    @pytest.mark.asyncio
    async def test_disabling_clears_state(self, monkeypatch):
        from services import invoice_reminder

        mock_set_enabled = AsyncMock()
        mock_clear = AsyncMock()
        with patch("services.monitor_state.set_monitor_enabled", mock_set_enabled), \
             patch.object(invoice_reminder, "_clear_state", mock_clear):
            await invoice_reminder.set_monitor_enabled("user1", False)
        mock_set_enabled.assert_awaited_once_with("invoice_reminder", "user1", False)
        mock_clear.assert_awaited_once_with("user1")

    @pytest.mark.asyncio
    async def test_enabling_does_not_clear_state(self, monkeypatch):
        from services import invoice_reminder

        mock_set_enabled = AsyncMock()
        mock_clear = AsyncMock()
        with patch("services.monitor_state.set_monitor_enabled", mock_set_enabled), \
             patch.object(invoice_reminder, "_clear_state", mock_clear):
            await invoice_reminder.set_monitor_enabled("user1", True)
        mock_clear.assert_not_called()


class TestRunOnceWorkHours:
    @pytest.mark.asyncio
    async def test_skips_outside_work_hours(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        from services import invoice_reminder

        off_hours = datetime(2026, 8, 20, 22, 0, tzinfo=ZoneInfo("Europe/Warsaw"))
        mock_poll = AsyncMock()
        with patch.object(invoice_reminder, "_poll_all_users", mock_poll):
            with patch("services.invoice_reminder.datetime") as mock_dt:
                mock_dt.now.return_value = off_hours
                await invoice_reminder.run_once()
        mock_poll.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_during_work_hours(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        from services import invoice_reminder

        in_hours = datetime(2026, 8, 20, 10, 0, tzinfo=ZoneInfo("Europe/Warsaw"))
        mock_poll = AsyncMock()
        with patch.object(invoice_reminder, "_poll_all_users", mock_poll):
            with patch("services.invoice_reminder.datetime") as mock_dt:
                mock_dt.now.return_value = in_hours
                await invoice_reminder.run_once()
        mock_poll.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_without_redis_url(self):
        from services import invoice_reminder

        mock_poll = AsyncMock()
        with patch.object(invoice_reminder, "_poll_all_users", mock_poll):
            await invoice_reminder.run_once()
        mock_poll.assert_not_called()


class TestHandleReply:
    @pytest.mark.asyncio
    async def test_no_pending_state_returns_none(self, monkeypatch):
        from services import invoice_reminder

        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=None)):
            result = await invoice_reminder.handle_reply("user1", "cokolwiek")
        assert result is None

    @pytest.mark.asyncio
    async def test_unrelated_reply_falls_through(self, monkeypatch):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"]}
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", AsyncMock(return_value=("unrelated", 0))):
            result = await invoice_reminder.handle_reply("user1", "jaka jest pogoda?")
        assert result is None

    @pytest.mark.asyncio
    async def test_issue_reply_calls_issue_all(self, monkeypatch):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1", "o2"]}
        mock_issue_all = AsyncMock(return_value="wystawione")
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", AsyncMock(return_value=("issue", 0))), \
             patch.object(invoice_reminder, "_issue_all", mock_issue_all):
            result = await invoice_reminder.handle_reply("user1", "tak wystaw")
        assert result == "wystawione"
        mock_issue_all.assert_awaited_once_with("user1", state)

    @pytest.mark.asyncio
    async def test_decline_reply_disables_monitor(self, monkeypatch):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"]}
        mock_set_enabled = AsyncMock()
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", AsyncMock(return_value=("decline", 0))), \
             patch.object(invoice_reminder, "set_monitor_enabled", mock_set_enabled):
            result = await invoice_reminder.handle_reply("user1", "przestań pytać")
        assert "wyłączyłem" in result.lower()
        mock_set_enabled.assert_awaited_once_with("user1", False)

    @pytest.mark.asyncio
    async def test_snooze_duration_reply_sets_snooze_and_confirms(self, monkeypatch):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"]}
        mock_set_snooze = AsyncMock()
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", AsyncMock(return_value=("snooze_duration", 180))), \
             patch.object(invoice_reminder, "_set_snooze", mock_set_snooze):
            result = await invoice_reminder.handle_reply("user1", "za 3 godziny")
        assert "3 godziny" in result
        mock_set_snooze.assert_awaited_once_with("user1", state, 180)

    @pytest.mark.asyncio
    async def test_snooze_duration_is_clamped_to_minimum(self, monkeypatch):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"]}
        mock_set_snooze = AsyncMock()
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", AsyncMock(return_value=("snooze_duration", 0))), \
             patch.object(invoice_reminder, "_set_snooze", mock_set_snooze):
            await invoice_reminder.handle_reply("user1", "za 0 minut")
        mock_set_snooze.assert_awaited_once_with("user1", state, invoice_reminder._MIN_SNOOZE_MINUTES)

    @pytest.mark.asyncio
    async def test_snooze_unspecified_asks_for_duration(self, monkeypatch):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"]}
        mock_await_duration = AsyncMock()
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", AsyncMock(return_value=("snooze_unspecified", 0))), \
             patch.object(invoice_reminder, "_await_duration", mock_await_duration):
            result = await invoice_reminder.handle_reply("user1", "później")
        assert "jak długo" in result.lower()
        mock_await_duration.assert_awaited_once_with("user1", state)
