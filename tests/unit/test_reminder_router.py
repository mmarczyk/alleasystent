"""Unit tests for services/reminder_router.py.

The point of this module is the tie: two reminders waiting at once, a bare
"tak", and nothing in the thread saying which one it answers. It must ASK
rather than pick — the invoice reminder's "yes" issues real VAT invoices.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


INVOICE_ASK = "🧾 Masz 1 niewystawioną fakturę dla już wysłanych zamówień: `123`.\n\nWystawić je teraz?"
MESSAGE_ASK = "💬 Masz 2 nieprzeczytanych wiadomości od kupujących (od: kowalski).\n\nPokazać je?"
OTHER_ASK = "Masz 3 nowe zamówienia do realizacji. Pokazać szczegóły?"

OPEN = {"status": "awaiting_response"}


class _Router:
    """Drives the router with both reminders stubbed, and an in-memory stand-in
    for the Redis-held "which one did you mean?" note."""

    def __init__(self, invoice_open: bool, message_open: bool):
        self.invoice_open = invoice_open
        self.message_open = message_open
        self.invoice = AsyncMock(return_value="INVOICE HANDLED")
        self.message = AsyncMock(return_value="MESSAGE HANDLED")
        self.ask: dict | None = None

    async def __call__(self, text, last_assistant_text=None):
        import services.reminder_router as rr

        async def _save(user_id, t, kinds):
            self.ask = {"text": t, "kinds": kinds}

        async def _clear(user_id):
            self.ask = None

        with patch("services.invoice_reminder.get_pending_state",
                   new=AsyncMock(return_value=OPEN if self.invoice_open else None)), \
             patch("services.message_reminder.get_pending_state",
                   new=AsyncMock(return_value=OPEN if self.message_open else None)), \
             patch("services.invoice_reminder.handle_reply", new=self.invoice), \
             patch("services.message_reminder.handle_reply", new=self.message), \
             patch.object(rr, "_load_ask", new=AsyncMock(side_effect=lambda u: self.ask)), \
             patch.object(rr, "_save_ask", new=AsyncMock(side_effect=_save)), \
             patch.object(rr, "_clear_ask", new=AsyncMock(side_effect=_clear)):
            return await rr.handle_reply("user-1", text, last_assistant_text)


class TestAmbiguousReplyAsksInsteadOfGuessing:
    async def test_bare_yes_with_both_open_and_no_thread_context_asks(self):
        router = _Router(invoice_open=True, message_open=True)
        out = await router("tak", None)
        assert "Nie mam pewności" in out
        assert "niewystawione faktury" in out and "nieprzeczytane wiadomości" in out
        router.invoice.assert_not_awaited()
        router.message.assert_not_awaited()

    async def test_the_ambiguous_wording_is_remembered_for_the_replay(self):
        router = _Router(invoice_open=True, message_open=True)
        await router("tak", None)
        assert router.ask["text"] == "tak"
        assert set(router.ask["kinds"]) == {"invoice_reminder", "message_reminder"}

    async def test_naming_one_replays_the_original_message_to_it(self):
        """"faktury" on its own is not an instruction — "tak" was."""
        router = _Router(invoice_open=True, message_open=True)
        await router("tak", None)
        out = await router("faktury", None)
        assert out == "INVOICE HANDLED"
        assert router.invoice.await_args.args[1] == "tak"
        router.message.assert_not_awaited()

    async def test_naming_the_other_one_routes_there_instead(self):
        router = _Router(invoice_open=True, message_open=True)
        await router("tak", None)
        out = await router("wiadomości", None)
        assert out == "MESSAGE HANDLED"
        assert router.message.await_args.args[1] == "tak"
        router.invoice.assert_not_awaited()

    async def test_answering_the_question_clears_it(self):
        router = _Router(invoice_open=True, message_open=True)
        await router("tak", None)
        await router("faktury", None)
        assert router.ask is None

    async def test_naming_both_does_not_trap_the_seller(self):
        """An unusable answer drops the question rather than looping on it."""
        router = _Router(invoice_open=True, message_open=True)
        await router("tak", None)
        out = await router("faktury i wiadomości", None)
        assert router.ask is None
        # Handed to normal routing instead of re-posing the same question to
        # someone who just tried to answer it — that would be a loop with no exit.
        assert out is None

    async def test_an_unrelated_message_afterwards_drops_the_question(self):
        router = _Router(invoice_open=True, message_open=True)
        await router("tak", None)
        await router("ile mam zamówień?", OTHER_ASK)
        assert router.ask is None


class TestUnambiguousRepliesStillGoStraightThrough:
    async def test_only_one_reminder_open_needs_no_question(self):
        router = _Router(invoice_open=True, message_open=False)
        assert await router("tak", None) == "INVOICE HANDLED"

    async def test_only_the_message_reminder_open(self):
        router = _Router(invoice_open=False, message_open=True)
        assert await router("tak", None) == "MESSAGE HANDLED"

    async def test_naming_the_topic_wins_even_with_both_open(self):
        router = _Router(invoice_open=True, message_open=True)
        assert await router("wystaw faktury", None) == "INVOICE HANDLED"
        router.message.assert_not_awaited()

    async def test_last_assistant_turn_settles_it_when_both_are_open(self):
        router = _Router(invoice_open=True, message_open=True)
        assert await router("tak", MESSAGE_ASK) == "MESSAGE HANDLED"
        router.invoice.assert_not_awaited()

    async def test_last_assistant_turn_settles_it_the_other_way(self):
        router = _Router(invoice_open=True, message_open=True)
        assert await router("tak", INVOICE_ASK) == "INVOICE HANDLED"
        router.message.assert_not_awaited()

    async def test_a_yes_for_a_different_question_reaches_neither(self):
        """The regression that once issued real VAT invoices."""
        router = _Router(invoice_open=True, message_open=True)
        assert await router("tak", OTHER_ASK) is None
        router.invoice.assert_not_awaited()
        router.message.assert_not_awaited()

    async def test_nothing_open_falls_through(self):
        router = _Router(invoice_open=False, message_open=False)
        assert await router("tak", None) is None

    async def test_duration_answer_goes_to_the_reminder_that_asked(self):
        """Both follow-ups have the same shape ("na jak długo…"), so the wording
        each uses must identify it — this is what the shared generic pattern
        used to get wrong."""
        from services.message_reminder import _ASK_DURATION_TEXT as MSG_DURATION

        router = _Router(invoice_open=True, message_open=True)
        assert await router("2 godziny", MSG_DURATION) == "MESSAGE HANDLED"
        router.invoice.assert_not_awaited()

    async def test_invoice_duration_answer_goes_to_the_invoice_reminder(self):
        from services.invoice_reminder import _ASK_DURATION_TEXT as INV_DURATION

        router = _Router(invoice_open=True, message_open=True)
        assert await router("2 godziny", INV_DURATION) == "INVOICE HANDLED"
        router.message.assert_not_awaited()


class TestOwnAskPatternsDoNotOverlap:
    """Each reminder must recognise only its OWN questions. When both matched
    the same text, whichever ran first silently took the other's reply."""

    def test_no_message_text_looks_like_an_invoice_question(self):
        from services.invoice_reminder import _OWN_ASK_RE as INV
        from services.message_reminder import _ASK_DURATION_TEXT as MSG_DURATION

        for text in (MESSAGE_ASK, MSG_DURATION):
            assert not INV.search(text), text

    def test_no_invoice_text_looks_like_a_message_question(self):
        from services.invoice_reminder import _ASK_DURATION_TEXT as INV_DURATION
        from services.message_reminder import _OWN_ASK_RE as MSG

        for text in (INVOICE_ASK, INV_DURATION):
            assert not MSG.search(text), text

    def test_each_still_recognises_its_own(self):
        from services.invoice_reminder import _ASK_DURATION_TEXT as INV_DURATION, _OWN_ASK_RE as INV
        from services.message_reminder import _ASK_DURATION_TEXT as MSG_DURATION, _OWN_ASK_RE as MSG

        assert INV.search(INVOICE_ASK) and INV.search(INV_DURATION)
        assert MSG.search(MESSAGE_ASK) and MSG.search(MSG_DURATION)
