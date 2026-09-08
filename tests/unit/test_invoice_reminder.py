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


def _order(order_id: str, *, company: str = "", first: str = "", last: str = "",
           login: str = "kupujacy", total: float = 100.0, currency: str = "PLN"):
    """An AllegroOrder as get_orders_needing_invoice returns it — the reminder
    reads the buyer and the value off the order it already has, no extra call."""
    from models.allegro import AllegroInvoiceBuyer, AllegroOrder

    return AllegroOrder(
        order_id=order_id,
        buyer_login=login,
        status="READY_FOR_PROCESSING",
        fulfillment_status="SENT",
        total_price=total,
        currency=currency,
        invoice_required=True,
        invoice_buyer=AllegroInvoiceBuyer(
            required=True, company_name=company, first_name=first, last_name=last,
        ),
    )


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

    def test_pending_invoice_phrase_uses_the_2_to_4_form(self):
        """"3 niewystawionych faktur" was the wrong Polish plural."""
        from services.invoice_reminder import _pending_invoice_phrase
        assert _pending_invoice_phrase(3) == "3 niewystawione faktury"
        assert _pending_invoice_phrase(5) == "5 niewystawionych faktur"
        assert _pending_invoice_phrase(12) == "12 niewystawionych faktur"
        assert _pending_invoice_phrase(22) == "22 niewystawione faktury"

    def test_count_phrase(self):
        from services.invoice_reminder import _count_phrase
        assert _count_phrase(1) == "1 fakturę"
        assert _count_phrase(3) == "3 faktury"
        assert _count_phrase(5) == "5 faktur"



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


class TestReminderIsChatOnly:
    """The reminder must read as the assistant writing in the chat, not as a
    system alert: no OS push, no entry in the notifications panel. The other
    monitors (orders, messages, returns) are unaffected and still notify."""

    async def _run_ask(self, **kwargs):
        from services import invoice_reminder, push_service

        store = AsyncMock()
        notify = AsyncMock()
        push = AsyncMock()
        with patch.object(push_service, "store_pending_chat", store), \
             patch.object(push_service, "add_notification", notify), \
             patch.object(push_service, "send_push", push):
            await invoice_reminder._ask("user1", [_order("o1"), _order("o2")], **kwargs)
        return store, notify, push

    @pytest.mark.asyncio
    async def test_first_ask_only_queues_a_chat_message(self):
        store, notify, push = await self._run_ask(again=False, awaiting_duration=False)
        store.assert_awaited_once()
        notify.assert_not_awaited()
        push.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_queued_under_a_dedupe_tag(self):
        """An unanswered reminder is replaced by the next one, not stacked."""
        from services.invoice_reminder import _MONITOR_KIND

        store, _, _ = await self._run_ask(again=True, awaiting_duration=False)
        assert store.await_args.kwargs["dedupe_tag"] == _MONITOR_KIND

    @pytest.mark.asyncio
    async def test_repeat_ask_mentions_the_orders(self):
        store, notify, push = await self._run_ask(again=True, awaiting_duration=True)
        text = store.await_args[0][1]
        assert "o1" in text and "o2" in text
        notify.assert_not_awaited()
        push.assert_not_awaited()


class TestReminderOwnsReply:
    """The reminder must not claim a message that answers a DIFFERENT question
    the assistant just asked. This was a real, critical bug: the assistant
    asked "Masz 1 nową wiadomość (od: Modelinarnia). Pokazać szczegóły?", the
    seller answered "Tak", and the open invoice reminder took that "Tak" as
    consent and issued 3 real VAT invoices."""

    ASK = "🧾 Masz 3 niewystawionych faktur dla już wysłanych zamówień: `o1`, `o2`, `o3`.\n\nWystawić je teraz?"
    OTHER_ASK = "Masz **1** nową wiadomość (od: Modelinarnia). Pokazać szczegóły?"

    def test_bare_yes_after_an_unrelated_question_is_not_the_reminders(self):
        from services.invoice_reminder import _reminder_owns_reply
        assert _reminder_owns_reply("Tak", self.OTHER_ASK) is False

    def test_bare_yes_after_the_reminders_own_ask_is_the_reminders(self):
        from services.invoice_reminder import _reminder_owns_reply
        assert _reminder_owns_reply("Tak", self.ASK) is True

    @pytest.mark.asyncio
    async def test_all_three_ask_variants_are_recognized(self):
        """Guards against the asks and the matcher drifting apart: whatever
        _ask actually sends must read back as the reminder's own question."""
        from services import invoice_reminder

        for again, awaiting in ((False, False), (True, False), (True, True)):
            with patch.object(invoice_reminder, "_notify", AsyncMock()) as notify:
                await invoice_reminder._ask(
                    "u1", [_order("o1"), _order("o2")],
                    again=again, awaiting_duration=awaiting,
                )
            text = notify.await_args.kwargs["chat_text"]
            assert invoice_reminder._reminder_owns_reply("tak", text) is True

    def test_duration_reply_after_the_how_long_follow_up_is_the_reminders(self):
        """"2 godziny" names no invoice, but it answers the reminder's own
        follow-up question — losing this would break the snooze flow."""
        from services.invoice_reminder import _ASK_DURATION_TEXT, _reminder_owns_reply
        assert _reminder_owns_reply("2 godziny", _ASK_DURATION_TEXT) is True

    def test_message_naming_invoices_is_the_reminders_whatever_was_asked(self):
        from services.invoice_reminder import _reminder_owns_reply
        assert _reminder_owns_reply("wystaw te faktury", self.OTHER_ASK) is True
        assert _reminder_owns_reply("przypomnij mi o fakturach jutro", self.OTHER_ASK) is True

    def test_no_assistant_turn_in_thread_keeps_the_cross_thread_case_working(self):
        """The reminder may be answered from a thread it was never written
        into (see the module docstring) — with nothing else to answer there,
        the reply is still the reminder's."""
        from services.invoice_reminder import _reminder_owns_reply
        assert _reminder_owns_reply("tak", None) is True
        assert _reminder_owns_reply("tak", "   ") is True


class TestHandleReplyRespectsOtherQuestions:
    OTHER_ASK = "Masz **1** nową wiadomość (od: Modelinarnia). Pokazać szczegóły?"

    @pytest.mark.asyncio
    async def test_yes_to_another_question_never_issues_invoices(self):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1", "o2", "o3"]}
        mock_issue_all = AsyncMock(return_value="Wystawiam 3 faktur:")
        mock_classify = AsyncMock(return_value=("issue", 0))
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", mock_classify), \
             patch.object(invoice_reminder, "_issue_all", mock_issue_all):
            result = await invoice_reminder.handle_reply("user1", "Tak", self.OTHER_ASK)

        assert result is None                 # falls through to normal routing
        mock_issue_all.assert_not_awaited()   # nothing was issued
        mock_classify.assert_not_awaited()    # not even classified

    @pytest.mark.asyncio
    async def test_yes_to_the_reminders_own_ask_still_issues(self):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"]}
        mock_issue_all = AsyncMock(return_value="wystawione")
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", AsyncMock(return_value=("issue", 0))), \
             patch.object(invoice_reminder, "_issue_all", mock_issue_all):
            result = await invoice_reminder.handle_reply(
                "user1", "Tak",
                "🧾 Masz 1 niewystawioną fakturę dla już wysłanych zamówień: `o1`.\n\nWystawić je teraz?",
            )
        assert result == "wystawione"
        mock_issue_all.assert_awaited_once_with("user1", state)

    @pytest.mark.asyncio
    async def test_last_assistant_turn_is_passed_to_the_classifier(self):
        """Second layer: even on a message the reminder may claim, the
        classifier is told what the assistant last asked."""
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"]}
        mock_classify = AsyncMock(return_value=("unrelated", 0))
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", mock_classify):
            await invoice_reminder.handle_reply("user1", "wystaw faktury", self.OTHER_ASK)
        mock_classify.assert_awaited_once_with("wystaw faktury", state, self.OTHER_ASK)


class TestReadOnlyQueriesNeverIssue:
    """A message that only asks to SEE the pending invoices must never issue
    them. This was a real, critical bug: the seller wrote "Pokaż mi faktury do
    wystawienia", the classifier called it ISSUE, and the reminder answered
    "Wystawiam 2 faktur:" — two irreversible VAT invoices for a question."""

    QUERIES = [
        "Pokaż mi faktury do wystawienia",
        "pokaz faktury do wystawienia",
        "wyświetl niewystawione faktury",
        "jakie mam faktury do wystawienia?",
        "które faktury czekają na wystawienie",
        "ile mam niewystawionych faktur",
        "sprawdź faktury do wystawienia",
        "lista faktur do wystawienia",
        "czy mam jakieś faktury do wystawienia",
        "a te faktury do wystawienia?",
    ]

    COMMANDS = [
        "tak",
        "wystaw",
        "wystaw je teraz",
        "ok, wystaw te faktury",
        "dawaj",
        "sprawdź i wystaw je",           # names both — the write wins, it was asked for
        "2 godziny",                     # duration reply must still reach the classifier
    ]

    def test_queries_are_read_only(self):
        from services.invoice_reminder import _is_read_only_query
        for text in self.QUERIES:
            assert _is_read_only_query(text) is True, text

    def test_commands_are_not_read_only(self):
        from services.invoice_reminder import _is_read_only_query
        for text in self.COMMANDS:
            assert _is_read_only_query(text) is False, text

    @pytest.mark.asyncio
    async def test_show_me_the_invoices_issues_nothing(self):
        """Even with the reminder wide open and the classifier saying ISSUE."""
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1", "o2"]}
        mock_issue_all = AsyncMock(return_value="Wystawiam 2 faktur:")
        mock_classify = AsyncMock(return_value=("issue", 0))
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", mock_classify), \
             patch.object(invoice_reminder, "_issue_all", mock_issue_all):
            result = await invoice_reminder.handle_reply(
                "user1", "Pokaż mi faktury do wystawienia", None,
            )

        assert result is None                 # falls through — the agent lists them
        mock_issue_all.assert_not_awaited()   # nothing was issued
        mock_classify.assert_not_awaited()    # not even classified

    @pytest.mark.asyncio
    async def test_show_me_after_the_reminders_own_ask_still_issues_nothing(self):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1", "o2"]}
        mock_issue_all = AsyncMock(return_value="Wystawiam 2 faktur:")
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", AsyncMock(return_value=("issue", 0))), \
             patch.object(invoice_reminder, "_issue_all", mock_issue_all):
            result = await invoice_reminder.handle_reply(
                "user1", "pokaż je najpierw",
                "🧾 Masz 2 niewystawionych faktur dla już wysłanych zamówień: `o1`, `o2`.\n\nWystawić je teraz?",
            )

        assert result is None
        mock_issue_all.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_explicit_issue_command_is_untouched(self):
        from services import invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"]}
        mock_issue_all = AsyncMock(return_value="wystawione")
        with patch.object(invoice_reminder, "get_pending_state", AsyncMock(return_value=state)), \
             patch.object(invoice_reminder, "_classify_reply", AsyncMock(return_value=("issue", 0))), \
             patch.object(invoice_reminder, "_issue_all", mock_issue_all):
            result = await invoice_reminder.handle_reply("user1", "ok, wystaw te faktury", None)
        assert result == "wystawione"
        mock_issue_all.assert_awaited_once_with("user1", state)


class TestRefreshPendingMessage:
    """A queued reminder waits in Redis until the seller opens the app — up to a
    day — so what it says about "invoices still to issue" can be plain wrong by
    the time they read it. This is what made the reminder point at orders whose
    invoice had meanwhile been issued (in Allegro's panel, from another device)."""

    @staticmethod
    def _allegro_returning(order_ids):
        allegro = AsyncMock()
        allegro._tokens = {"access_token": "t"}
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_orders_needing_invoice = AsyncMock(
            return_value=[_order(oid) for oid in order_ids]
        )
        return allegro

    @pytest.mark.asyncio
    async def test_no_open_ask_drops_the_queued_message(self, monkeypatch):
        from services import invoice_reminder

        monkeypatch.setattr(invoice_reminder, "get_pending_state", AsyncMock(return_value=None))
        assert await invoice_reminder.refresh_pending_message("user1", "🧾 Masz 1 …") is None

    @pytest.mark.asyncio
    async def test_all_invoices_issued_since_drops_the_message(self, monkeypatch):
        from services import allegro_service, invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"], "reminder_count": 1}
        resolve = AsyncMock()
        monkeypatch.setattr(invoice_reminder, "get_pending_state", AsyncMock(return_value=state))
        monkeypatch.setattr(invoice_reminder, "_resolve_state", resolve)
        monkeypatch.setattr(
            allegro_service.AllegroService, "get_instance",
            lambda user_id: self._allegro_returning([]),
        )

        assert await invoice_reminder.refresh_pending_message("user1", "🧾 Masz 1 …") is None
        resolve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unchanged_pending_set_shows_the_queued_text(self, monkeypatch):
        from services import allegro_service, invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1", "o2"], "reminder_count": 1}
        monkeypatch.setattr(invoice_reminder, "get_pending_state", AsyncMock(return_value=state))
        monkeypatch.setattr(
            allegro_service.AllegroService, "get_instance",
            lambda user_id: self._allegro_returning(["o2", "o1"]),
        )

        text = "🧾 Masz 2 niewystawionych faktur …"
        assert await invoice_reminder.refresh_pending_message("user1", text) == text

    @pytest.mark.asyncio
    async def test_shrunken_pending_set_is_rewritten_and_state_updated(self, monkeypatch):
        from services import allegro_service, invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1", "o2"], "reminder_count": 1}
        update = AsyncMock()
        monkeypatch.setattr(invoice_reminder, "get_pending_state", AsyncMock(return_value=state))
        monkeypatch.setattr(invoice_reminder, "_update_pending_orders", update)
        monkeypatch.setattr(
            allegro_service.AllegroService, "get_instance",
            lambda user_id: self._allegro_returning(["o2"]),
        )

        text = await invoice_reminder.refresh_pending_message("user1", "🧾 Masz 2 …")
        assert "o2" in text and "o1" not in text
        assert "1 niewystawioną fakturę" in text
        # The reply ("wystaw") must act on what the seller was actually shown.
        assert update.await_args[0][2] == ["o2"]

    @pytest.mark.asyncio
    async def test_allegro_failure_still_delivers_the_original(self, monkeypatch):
        from services import allegro_service, invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"], "reminder_count": 1}
        allegro = AsyncMock()
        allegro._tokens = {"access_token": "t"}
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_orders_needing_invoice = AsyncMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(invoice_reminder, "get_pending_state", AsyncMock(return_value=state))
        monkeypatch.setattr(
            allegro_service.AllegroService, "get_instance", lambda user_id: allegro
        )

        text = "🧾 Masz 1 niewystawioną fakturę …"
        assert await invoice_reminder.refresh_pending_message("user1", text) == text


class TestSellerSaysItIsAlreadyIssued:
    """Allegro is the only thing that decides whether an invoice is missing, so
    "przecież ta faktura już jest" is not something to write down and believe —
    it is a reason to go and ask Allegro again, right now."""

    def test_classification_is_parsed(self):
        from services.invoice_reminder import _parse_classification

        assert _parse_classification("ALREADY_ISSUED") == ("already_issued", 0)

    def test_is_not_confused_with_an_issue_command(self):
        from services.invoice_reminder import _parse_classification

        assert _parse_classification("ISSUE") == ("issue", 0)
        assert _parse_classification("ALREADY_ISSUED")[0] != "issue"

    @staticmethod
    def _allegro_returning(order_ids):
        allegro = AsyncMock()
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_orders_needing_invoice = AsyncMock(
            return_value=[_order(oid) for oid in order_ids]
        )
        return allegro

    @pytest.mark.asyncio
    async def test_allegro_agreeing_stops_the_reminder_and_issues_nothing(self, monkeypatch):
        from services import allegro_service, invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"], "interval_minutes": 120}
        resolve, issue = AsyncMock(), AsyncMock()
        monkeypatch.setattr(invoice_reminder, "_resolve_state", resolve)
        monkeypatch.setattr(invoice_reminder, "_issue_all", issue)
        monkeypatch.setattr(invoice_reminder, "get_pending_state", AsyncMock(return_value=state))
        monkeypatch.setattr(
            invoice_reminder, "_classify_reply", AsyncMock(return_value=("already_issued", 0))
        )
        monkeypatch.setattr(
            allegro_service.AllegroService, "get_instance",
            lambda user_id: self._allegro_returning([]),
        )

        out = await invoice_reminder.handle_reply("user1", "przecież ta faktura już jest", None)

        assert "masz rację" in out.lower()
        resolve.assert_awaited_once()
        issue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allegro_disagreeing_says_what_is_missing_and_goes_quiet(self, monkeypatch):
        """Never "you're wrong, shall I issue another?" — the invoice may well
        exist; what Allegro lacks is the PDF on the order."""
        from services import allegro_service, invoice_reminder

        state = {"status": "awaiting_response", "order_ids": ["o1"], "interval_minutes": 120}
        snooze, issue = AsyncMock(), AsyncMock()
        monkeypatch.setattr(invoice_reminder, "_set_snooze", snooze)
        monkeypatch.setattr(invoice_reminder, "_issue_all", issue)
        monkeypatch.setattr(invoice_reminder, "get_pending_state", AsyncMock(return_value=state))
        monkeypatch.setattr(
            invoice_reminder, "_classify_reply", AsyncMock(return_value=("already_issued", 0))
        )
        monkeypatch.setattr(
            allegro_service.AllegroService, "get_instance",
            lambda user_id: self._allegro_returning(["o1"]),
        )

        out = await invoice_reminder.handle_reply("user1", "ta faktura już jest", None)

        assert "dołącz fakturę do zamówienia" in out
        assert "o1" in out
        issue.assert_not_awaited()
        assert snooze.await_args[0][2] == invoice_reminder._RECHECK_SNOOZE_MINUTES

    @pytest.mark.asyncio
    async def test_a_failed_recheck_says_so_instead_of_deciding(self, monkeypatch):
        from services import allegro_service, invoice_reminder
        from services.allegro_service import AllegroAPIError

        state = {"status": "awaiting_response", "order_ids": ["o1"], "interval_minutes": 120}
        allegro = AsyncMock()
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_orders_needing_invoice = AsyncMock(side_effect=AllegroAPIError(500, "boom"))
        monkeypatch.setattr(invoice_reminder, "get_pending_state", AsyncMock(return_value=state))
        monkeypatch.setattr(
            invoice_reminder, "_classify_reply", AsyncMock(return_value=("already_issued", 0))
        )
        monkeypatch.setattr(
            allegro_service.AllegroService, "get_instance", lambda user_id: allegro
        )

        out = await invoice_reminder.handle_reply("user1", "faktura już jest", None)
        assert "Nie udało mi się" in out


class TestRemindingIsAllegrosCallAlone:
    """The reminder must never suppress itself from internal state — an invoice
    can be attached to an order by anyone at any moment, and only Allegro knows."""

    def test_the_reminder_does_not_consult_the_issuance_ledger(self):
        import inspect

        from services import invoice_reminder

        source = inspect.getsource(invoice_reminder)
        assert "invoice_ledger" not in source

class TestOrderLines:
    """An order id identifies nothing to a human — the seller could not tell
    from the old reminder whose invoice was missing, or whether it was worth
    30 zł or 3000 zł, which is what decides whether they deal with it now."""

    def test_a_company_buyer_is_named_with_the_value(self):
        from services.invoice_reminder import _format_order_lines

        line = _format_order_lines([_order("o1", company="Modelinarnia sp. z o.o.", total=249.0)])

        assert "Modelinarnia sp. z o.o." in line
        assert "249,00 PLN" in line
        assert "`o1`" in line

    def test_a_private_buyer_is_named_from_the_invoice_address(self):
        from services.invoice_reminder import _format_order_lines

        line = _format_order_lines([_order("o1", first="Jan", last="Kowalski")])

        assert "Jan Kowalski" in line

    def test_the_allegro_login_is_only_the_fallback(self):
        """A login like "kot123" says nothing — but it beats naming nobody."""
        from services.invoice_reminder import _format_order_lines

        assert "kot123" in _format_order_lines([_order("o1", login="kot123")])

    def test_a_total_is_added_for_more_than_one_order(self):
        from services.invoice_reminder import _format_order_lines

        out = _format_order_lines([
            _order("o1", company="Firma A", total=100.0),
            _order("o2", company="Firma B", total=49.5),
        ])

        assert "Razem" in out and "149,50 PLN" in out

    def test_a_single_order_gets_no_total(self):
        from services.invoice_reminder import _format_order_lines

        assert "Razem" not in _format_order_lines([_order("o1", total=100.0)])

    def test_mixed_currencies_are_not_summed(self):
        from services.invoice_reminder import _format_order_lines

        out = _format_order_lines([
            _order("o1", total=100.0, currency="PLN"),
            _order("o2", total=50.0, currency="EUR"),
        ])

        assert "Razem" not in out

    def test_the_ask_itself_carries_the_buyer_and_the_value(self):
        """The whole point: the seller reads the reminder, not the order ids."""
        from services.invoice_reminder import _build_ask_text

        text = _build_ask_text(
            [_order("o1", company="Modelinarnia sp. z o.o.", total=249.0)],
            again=False, awaiting_duration=False,
        )

        assert "Modelinarnia sp. z o.o." in text
        assert "249,00 PLN" in text
        assert "`o1`" in text

    def test_the_listing_truncates(self):
        from services.invoice_reminder import _format_order_lines

        out = _format_order_lines([_order(f"order-{i}", total=1.0) for i in range(12)])

        assert "order-0" in out
        assert "order-9" in out
        assert "order-10" not in out
        assert "2 więcej" in out
