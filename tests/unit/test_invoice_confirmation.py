"""The seller confirms an invoice before it leaves for Allegro or KSeF.

Issuing an invoice used to attach it to the Allegro order in the same breath,
so a real, numbered VAT invoice reached the buyer's order page before anyone
had looked at it — and a "wystaw i wyślij do KSeF" could have gone straight on
to the tax authority, which is not something to take back. Both deliveries now
wait for the seller to say the invoice is correct, and a confirmation is
necessarily a NEW message: these tests pin that the delivery tools refuse to
run in the same turn as the issuance, and that a later turn goes through.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.allegro.allegro_agent import AllegroAgent


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("INFAKT_API_KEY", "test-infakt-key")
    # No Redis: the invoice ledger degrades to "nothing recorded", which is the
    # harder case here — the gate must not depend on it.
    monkeypatch.delenv("REDIS_URL", raising=False)


ORDER_ID = "ORD-1"
INVOICE_UUID = "inv-9"


def _agent():
    with patch("agents.base_agent.AsyncOpenAI"), \
         patch("agents.allegro.allegro_agent.AllegroService") as MockService:
        service = MagicMock()
        service._user_id = "u1"
        service._tokens = MagicMock()
        service._tokens.is_expired.return_value = False
        service.has_scope.return_value = True
        service.get_order = AsyncMock(return_value=type(
            "O", (), {"invoice_required": True, "buyer_login": "kupujacy"}
        )())
        service.get_order_invoices = AsyncMock(return_value=[])
        service.get_order_invoice_data = AsyncMock(return_value={"company_name": "Firma"})
        service.create_order_invoice_record = AsyncMock(return_value="allegro-inv-1")
        service.upload_order_invoice_file = AsyncMock()
        MockService.get_instance.return_value = service
        return AllegroAgent()


def _infakt():
    infakt = MagicMock()
    infakt.create_invoice = AsyncMock(return_value={"invoice_uuid": INVOICE_UUID})
    infakt.get_share_link = AsyncMock(return_value="https://app.infakt.pl/share/inv-9")
    infakt.get_invoice = AsyncMock(return_value={"number": "FV/1/2026"})
    infakt.get_invoice_pdf = AsyncMock(return_value=b"%PDF-1.4 tiny")
    infakt.send_to_ksef = AsyncMock(return_value={"status": "sent"})
    return infakt


def _wired(infakt):
    return patch(
        "services.infakt_service.InfaktService.get_instance", return_value=infakt
    )


class TestIssuanceDeliversNothingByItself:
    async def test_issuing_does_not_touch_allegro(self):
        agent, infakt = _agent(), _infakt()

        with _wired(infakt), patch("services.infakt_service.build_invoice_payload", return_value={}):
            out = await agent._dispatch("issue_invoice_for_order", {"order_id": ORDER_ID})

        agent._allegro.create_order_invoice_record.assert_not_awaited()
        agent._allegro.upload_order_invoice_file.assert_not_awaited()
        infakt.send_to_ksef.assert_not_awaited()
        assert "NIE dołączyłem" in out
        assert f"`{INVOICE_UUID}`" in out


class TestSameTurnDeliveryIsRefused:
    """The model chaining issue → attach in one turn is exactly the behaviour
    being removed; the tool refuses it rather than trusting the prompt."""

    async def _issue(self, agent, infakt):
        with _wired(infakt), patch("services.infakt_service.build_invoice_payload", return_value={}):
            await agent._dispatch("issue_invoice_for_order", {"order_id": ORDER_ID})

    async def test_attaching_right_after_issuing_is_refused(self):
        agent, infakt = _agent(), _infakt()
        await self._issue(agent, infakt)

        with _wired(infakt):
            out = await agent._dispatch(
                "attach_invoice_to_allegro_order",
                {"order_id": ORDER_ID, "invoice_uuid": INVOICE_UUID},
            )

        agent._allegro.create_order_invoice_record.assert_not_awaited()
        agent._allegro.upload_order_invoice_file.assert_not_awaited()
        assert "potwierdzenia" in out
        assert "dołącz fakturę do zamówienia" in out

    async def test_ksef_right_after_issuing_is_refused(self):
        agent, infakt = _agent(), _infakt()
        await self._issue(agent, infakt)

        with _wired(infakt):
            out = await agent._dispatch("send_invoice_to_ksef", {"invoice_uuid": INVOICE_UUID})

        infakt.send_to_ksef.assert_not_awaited()
        assert "potwierdzenia" in out
        assert "KSeF" in out

    async def test_the_order_id_alone_is_enough_to_recognize_the_same_turn(self):
        """A timed-out issuance leaves no invoice id — the order still must not
        have something attached to it on the same breath."""
        agent, infakt = _agent(), _infakt()
        infakt.create_invoice.side_effect = TimeoutError("still pending")
        await self._issue(agent, infakt)

        with _wired(infakt):
            out = await agent._dispatch(
                "attach_invoice_to_allegro_order",
                {"order_id": ORDER_ID, "invoice_uuid": "guessed-uuid"},
            )

        agent._allegro.create_order_invoice_record.assert_not_awaited()
        assert "potwierdzenia" in out


class TestAConfirmedDeliveryGoesThrough:
    """The gate is about THIS turn only — the seller coming back to say "ok" is
    a new turn, and then the invoice really does go out."""

    async def test_attaching_in_a_later_turn_works(self):
        agent, infakt = _agent(), _infakt()

        with _wired(infakt):
            out = await agent._dispatch(
                "attach_invoice_to_allegro_order",
                {"order_id": ORDER_ID, "invoice_uuid": INVOICE_UUID},
            )

        agent._allegro.create_order_invoice_record.assert_awaited_once()
        agent._allegro.upload_order_invoice_file.assert_awaited_once()
        assert "✅" in out

    async def test_ksef_in_a_later_turn_works(self):
        agent, infakt = _agent(), _infakt()

        with _wired(infakt):
            out = await agent._dispatch("send_invoice_to_ksef", {"invoice_uuid": INVOICE_UUID})

        infakt.send_to_ksef.assert_awaited_once_with(INVOICE_UUID)
        assert "📤" in out

    async def test_a_new_turn_clears_what_the_previous_one_issued(self):
        """run() resets the gate — the agent instance is cached per user in the
        orchestrator, so without this a confirmation would be refused for ever."""
        agent = _agent()
        agent._issued_this_turn = {ORDER_ID, INVOICE_UUID}
        agent._allegro._tokens = None
        agent._allegro._load_tokens_from_redis = AsyncMock()

        await agent.run("Dołącz tę fakturę do zamówienia")

        assert agent._issued_this_turn == set()
