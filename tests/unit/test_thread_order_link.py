"""Unit tests for pulling an order number out of a buyer's message.

A buyer writes "czy jest jeszcze możliwość wystawienia faktury do tej
transakcji" and names no transaction anywhere in the sentence. The order number
is in the message all the same — Allegro tags a message written from an order
page with `relatedObject: {"type": "ORDER", "id": ...}` — and where it isn't,
the buyer's own order history usually answers it.

These tests pin both halves: the tag is read (and an offer/dispute tag is
never mistaken for an order), a single order in the history counts as an
identification while several stay a question, and none of it can cost the
seller the message text they actually asked to read.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.allegro import AllegroOrder, AllegroOrderLine
from services.allegro_service import (
    AllegroAPIError,
    AllegroService,
    message_related_order_id,
    thread_buyer_login,
    thread_related_order_id,
)


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


ORDER_A = "0c4854a0-9646-11f1-8028-338c43adc37a"
ORDER_B = "1d5965b1-a757-22e2-9139-449d54bed48b"
BUYER = "andziacz2"


def _msg(text: str, *, at: str, login: str = BUYER, interlocutor: bool = True, **extra) -> dict:
    return {
        "id": f"m-{at}",
        "createdAt": at,
        "author": {"login": login, "isInterlocutor": interlocutor},
        "text": text,
        **extra,
    }


def _order(order_id: str, *, name: str = "Ekspres DeLonghi", total: float = 429.98) -> AllegroOrder:
    return AllegroOrder(
        order_id=order_id,
        buyer_login=BUYER,
        status="READY_FOR_PROCESSING",
        total_price=total,
        created_at="2026-09-01T10:00:00Z",
        line_items=[AllegroOrderLine(offer_id="1", offer_name=name, quantity=1, price=total)],
    )


def _service() -> AllegroService:
    """The resolver touches nothing but three API methods, all stubbed here —
    no OAuth, no HTTP client, no token file."""
    service = object.__new__(AllegroService)
    service.get_thread_messages = AsyncMock(return_value=[])
    service.get_order = AsyncMock(side_effect=AssertionError("unexpected get_order"))
    service.get_orders = AsyncMock(side_effect=AssertionError("unexpected get_orders"))
    return service


class TestMessageRelatedOrderId:
    def test_order_tag_is_the_order_number(self):
        assert message_related_order_id(
            _msg("faktura?", at="1", relatedObject={"type": "ORDER", "id": ORDER_A})
        ) == ORDER_A

    def test_offer_and_dispute_tags_are_not_order_numbers(self):
        """The id in an OFFER tag is an offer id. Reading it as a checkout form
        would attach an invoice to a form that doesn't exist — or worse, one
        that belongs to somebody else."""
        for kind in ("OFFER", "DISPUTE"):
            assert message_related_order_id(
                _msg("pytanie", at="1", relatedObject={"type": kind, "id": "14587236901"})
            ) == ""

    def test_untagged_message_has_no_order_number(self):
        assert message_related_order_id(_msg("Witam, czy wysyłacie dziś?", at="1")) == ""
        assert message_related_order_id(_msg("x", at="1", relatedObject=None)) == ""

    def test_nested_order_object_is_read_too(self):
        assert message_related_order_id(_msg("x", at="1", order={"id": ORDER_B})) == ORDER_B

    def test_type_is_matched_case_insensitively(self):
        assert message_related_order_id(
            _msg("x", at="1", relatedObject={"type": "order", "id": ORDER_A})
        ) == ORDER_A


class TestThreadLevelHelpers:
    def test_newest_tagged_message_wins(self):
        """A thread reused for a second purchase asks about the second one."""
        messages = [
            _msg("pierwsza", at="2026-09-01T10:00:00Z",
                 relatedObject={"type": "ORDER", "id": ORDER_A}),
            _msg("druga", at="2026-09-08T18:42:00Z",
                 relatedObject={"type": "ORDER", "id": ORDER_B}),
        ]

        assert thread_related_order_id(messages) == ORDER_B
        assert thread_related_order_id(list(reversed(messages))) == ORDER_B

    def test_untagged_thread_has_no_order(self):
        assert thread_related_order_id([_msg("cześć", at="1")]) == ""
        assert thread_related_order_id([]) == ""

    def test_buyer_login_comes_from_the_interlocutor(self):
        messages = [
            _msg("Dzień dobry", at="1", login="sklep_agd", interlocutor=False),
            _msg("Witam", at="2"),
        ]

        assert thread_buyer_login(messages) == BUYER
        assert thread_buyer_login([_msg("x", at="1", login="sklep_agd", interlocutor=False)]) == ""


class TestResolveThreadOrder:
    @pytest.mark.asyncio
    async def test_tagged_message_resolves_exactly(self):
        service = _service()
        service.get_order = AsyncMock(return_value=_order(ORDER_A))

        match = await service.resolve_thread_order(
            "t1",
            messages=[_msg("faktura?", at="1", relatedObject={"type": "ORDER", "id": ORDER_A})],
        )

        assert (match.order_id, match.source, match.resolved) == (ORDER_A, "message", True)
        assert match.buyer_login == BUYER
        assert [o.order_id for o in match.candidates] == [ORDER_A]
        service.get_orders.assert_not_called()

    @pytest.mark.asyncio
    async def test_tag_survives_an_unreadable_order(self):
        """A 403 or a 5xx on the order details loses the details, not the number
        — the seller can still act on the id."""
        service = _service()
        service.get_order = AsyncMock(side_effect=AllegroAPIError(403, "brak uprawnień"))

        match = await service.resolve_thread_order(
            "t1",
            messages=[_msg("faktura?", at="1", relatedObject={"type": "ORDER", "id": ORDER_A})],
        )

        assert (match.order_id, match.source) == (ORDER_A, "message")
        assert match.candidates == []

    @pytest.mark.asyncio
    async def test_untagged_message_falls_back_to_a_single_order(self):
        service = _service()
        service.get_orders = AsyncMock(return_value=[_order(ORDER_A)])

        match = await service.resolve_thread_order(
            "t1", messages=[_msg("czy mogę dostać fakturę?", at="1")]
        )

        assert (match.order_id, match.source) == (ORDER_A, "buyer_history")
        service.get_orders.assert_awaited_once()
        assert service.get_orders.await_args.kwargs["buyer_login"] == BUYER

    @pytest.mark.asyncio
    async def test_several_orders_stay_a_question(self):
        """Two candidate transactions and no tag is not an answer — picking one
        would attach an invoice to the wrong purchase."""
        service = _service()
        service.get_orders = AsyncMock(return_value=[_order(ORDER_B), _order(ORDER_A)])

        match = await service.resolve_thread_order(
            "t1", messages=[_msg("czy mogę dostać fakturę?", at="1")]
        )

        assert match.order_id == ""
        assert match.resolved is False
        assert match.source == ""
        assert [o.order_id for o in match.candidates] == [ORDER_B, ORDER_A]

    @pytest.mark.asyncio
    async def test_unreadable_history_is_not_an_error(self):
        service = _service()
        service.get_orders = AsyncMock(side_effect=AllegroAPIError(429, "za dużo żądań"))

        match = await service.resolve_thread_order("t1", messages=[_msg("hej", at="1")])

        assert (match.order_id, match.buyer_login, match.candidates) == ("", BUYER, [])

    @pytest.mark.asyncio
    async def test_no_buyer_no_lookup(self):
        service = _service()

        match = await service.resolve_thread_order("t1", messages=[])

        assert match.order_id == "" and match.buyer_login == ""
        service.get_orders.assert_not_called()

    @pytest.mark.asyncio
    async def test_messages_are_reused_not_refetched(self):
        """get_thread_messages has already been called by the time this runs —
        passing its result in keeps the tag lookup free."""
        service = _service()
        service.get_order = AsyncMock(return_value=_order(ORDER_A))

        await service.resolve_thread_order(
            "t1",
            messages=[_msg("x", at="1", relatedObject={"type": "ORDER", "id": ORDER_A})],
        )

        service.get_thread_messages.assert_not_called()

    @pytest.mark.asyncio
    async def test_messages_are_fetched_when_not_supplied(self):
        service = _service()
        service.get_thread_messages = AsyncMock(
            return_value=[_msg("x", at="1", relatedObject={"type": "ORDER", "id": ORDER_A})]
        )
        service.get_order = AsyncMock(return_value=_order(ORDER_A))

        match = await service.resolve_thread_order("t1")

        assert match.order_id == ORDER_A
        service.get_thread_messages.assert_awaited_once_with("t1")


def _agent():
    from agents.allegro.allegro_agent import AllegroAgent

    with patch("agents.base_agent.AsyncOpenAI"), \
         patch("agents.allegro.allegro_agent.AllegroService") as MockService:
        mock_service = MagicMock()
        mock_service._tokens = MagicMock()
        mock_service._tokens.is_expired.return_value = False
        MockService.get_instance.return_value = mock_service
        agent = AllegroAgent()
    agent._allegro.get_thread_messages = AsyncMock(
        return_value=[_msg("Witam, czy jest jeszcze możliwość wystawienia faktury?",
                           at="2026-09-09T18:42:00Z")]
    )
    return agent


class TestThreadMessagesShowTheOrder:
    """The rendered thread view is what the seller reads and what the next turn
    remembers (history carries the rendered text, never tool arguments), so the
    order number has to be IN it — not just resolved somewhere."""

    @pytest.mark.asyncio
    async def test_tagged_thread_names_the_order_under_the_message(self):
        agent = _agent()
        agent._allegro.resolve_thread_order = AsyncMock(return_value=MagicMock(
            source="message", order_id=ORDER_A, buyer_login=BUYER,
            candidates=[_order(ORDER_A)],
        ))

        out = await agent._dispatch("get_thread_messages", {"thread_id": "t1"})

        assert "Zamówienie z tej wiadomości" in out
        assert ORDER_A in out
        assert "Ekspres DeLonghi" in out
        # The message itself is still there, verbatim.
        assert "czy jest jeszcze możliwość wystawienia faktury?" in out

    @pytest.mark.asyncio
    async def test_untagged_thread_lists_the_candidates_instead_of_guessing(self):
        agent = _agent()
        agent._allegro.resolve_thread_order = AsyncMock(return_value=MagicMock(
            source="", order_id="", buyer_login=BUYER,
            candidates=[_order(ORDER_A), _order(ORDER_B, name="Czajnik Bosch")],
        ))

        out = await agent._dispatch("get_thread_messages", {"thread_id": "t1"})

        assert "nie ma podpiętego zamówienia" in out.lower()
        assert ORDER_A in out and ORDER_B in out
        assert "2 zamówienia" in out

    @pytest.mark.asyncio
    async def test_single_order_fallback_says_where_the_number_came_from(self):
        agent = _agent()
        agent._allegro.resolve_thread_order = AsyncMock(return_value=MagicMock(
            source="buyer_history", order_id=ORDER_A, buyer_login=BUYER,
            candidates=[_order(ORDER_A)],
        ))

        out = await agent._dispatch("get_thread_messages", {"thread_id": "t1"})

        assert ORDER_A in out
        assert "jedyne zamówienie konta" in out
        assert BUYER in out

    @pytest.mark.asyncio
    async def test_a_failed_lookup_never_costs_the_message(self):
        agent = _agent()
        agent._allegro.resolve_thread_order = AsyncMock(
            side_effect=AllegroAPIError(500, "Allegro niedostępne")
        )

        out = await agent._dispatch("get_thread_messages", {"thread_id": "t1"})

        assert "czy jest jeszcze możliwość wystawienia faktury?" in out
        assert "Zamówienie z tej wiadomości" not in out

    @pytest.mark.asyncio
    async def test_placeholder_buyer_is_not_used_as_a_login(self):
        """_dispatch labels an unnamed interlocutor "N/A"; looking orders up by
        that string finds nothing and hides the real fallback."""
        agent = _agent()
        agent._allegro.get_message_threads = AsyncMock(
            return_value=[{"id": "t1", "read": False, "interlocutor": {},
                           "lastMessageDateTime": "2026-09-09T16:42:00Z"}]
        )
        agent._allegro.resolve_thread_order = AsyncMock(return_value=MagicMock(
            source="", order_id="", buyer_login=BUYER, candidates=[],
        ))

        await agent._dispatch("get_thread_messages", {"date": "2026-09-09"})

        assert agent._allegro.resolve_thread_order.await_args.kwargs["buyer_login"] == ""
