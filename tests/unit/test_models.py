"""Unit tests for models/allegro.py and models/conversation.py."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from models.allegro import (
    AllegroAddress,
    AllegroMessage,
    AllegroOrder,
    AllegroOrderLine,
    AllegroTokens,
    DeliveryStatus,
    OrderStatus,
)
from models.conversation import (
    ChannelType,
    ConversationSession,
    MessageRole,
)


class TestAllegroTokensIsExpired:
    def _make_tokens(self, expires_at: datetime) -> AllegroTokens:
        return AllegroTokens(
            access_token="acc",
            refresh_token="ref",
            expires_at=expires_at,
        )

    def test_not_expired_when_future(self):
        future = datetime.utcnow() + timedelta(hours=1)
        assert self._make_tokens(future).is_expired() is False

    def test_expired_when_past(self):
        past = datetime.utcnow() - timedelta(seconds=1)
        assert self._make_tokens(past).is_expired() is True

    def test_expired_at_exact_boundary(self):
        now = datetime(2025, 1, 1, 12, 0, 0)
        tokens = self._make_tokens(now)
        with patch("models.allegro.datetime") as mock_dt:
            mock_dt.utcnow.return_value = now
            assert tokens.is_expired() is True

    def test_default_token_type(self):
        tokens = self._make_tokens(datetime.utcnow() + timedelta(hours=1))
        assert tokens.token_type == "Bearer"


class TestEnums:
    def test_order_status_values(self):
        assert OrderStatus.BOUGHT == "BOUGHT"
        assert OrderStatus.FILLED_IN == "FILLED_IN"
        assert OrderStatus.READY_FOR_PROCESSING == "READY_FOR_PROCESSING"
        assert OrderStatus.CANCELLED == "CANCELLED"

    def test_delivery_status_values(self):
        assert DeliveryStatus.WAITING == "WAITING"
        assert DeliveryStatus.IN_TRANSIT == "IN_TRANSIT"
        assert DeliveryStatus.DELIVERED == "DELIVERED"
        assert DeliveryStatus.FAILED == "FAILED"

    def test_order_status_is_str(self):
        assert isinstance(OrderStatus.BOUGHT, str)

    def test_delivery_status_is_str(self):
        assert isinstance(DeliveryStatus.WAITING, str)


class TestAllegroAddress:
    def test_defaults(self):
        addr = AllegroAddress()
        assert addr.first_name == ""
        assert addr.country_code == "PL"

    def test_full_construction(self):
        addr = AllegroAddress(
            first_name="Jan",
            last_name="Kowalski",
            street="ul. Testowa 1",
            city="Warszawa",
            zip_code="00-001",
            country_code="PL",
            phone_number="+48500000000",
        )
        assert addr.city == "Warszawa"
        assert addr.phone_number == "+48500000000"


class TestAllegroOrderLine:
    def test_construction(self):
        line = AllegroOrderLine(
            offer_id="offer-1",
            offer_name="Product A",
            quantity=3,
            price=99.99,
        )
        assert line.currency == "PLN"
        assert line.quantity == 3
        assert line.price == pytest.approx(99.99)


class TestAllegroOrderCoerceDelivery:
    def _base_kwargs(self, delivery):
        return dict(
            order_id="ord-1",
            buyer_login="buyer",
            status="BOUGHT",
            delivery=delivery,
        )

    def test_dict_delivery_preserved(self):
        order = AllegroOrder(**self._base_kwargs({"method": "InPost"}))
        assert order.delivery == {"method": "InPost"}

    def test_none_delivery_coerced_to_empty_dict(self):
        order = AllegroOrder(**self._base_kwargs(None))
        assert order.delivery == {}

    def test_string_delivery_coerced_to_empty_dict(self):
        order = AllegroOrder(**self._base_kwargs("some string"))
        assert order.delivery == {}

    def test_defaults(self):
        order = AllegroOrder(
            order_id="o1",
            buyer_login="buyer",
            status="BOUGHT",
        )
        assert order.total_price == 0.0
        assert order.invoice_required is False
        assert order.line_items == []


class TestAllegroMessage:
    def test_defaults(self):
        msg = AllegroMessage(thread_id="t1", text="hello")
        assert msg.type == "QUERY"
        assert msg.author_login == ""
        assert msg.created_at == ""


class TestConversationSession:
    def _make_session(self):
        return ConversationSession(
            session_id="s1",
            channel=ChannelType.API,
            sender_id="user1",
        )

    def test_add_message_appends(self):
        session = self._make_session()
        session.add_message(MessageRole.USER, "hello")
        assert len(session.messages) == 1
        assert session.messages[0].content == "hello"
        assert session.messages[0].role == MessageRole.USER

    def test_add_message_updates_updated_at(self):
        session = self._make_session()
        original_time = session.updated_at
        import time; time.sleep(0.01)
        session.add_message(MessageRole.USER, "ping")
        assert session.updated_at >= original_time

    def test_to_anthropic_messages_filters_system(self):
        session = self._make_session()
        session.add_message(MessageRole.SYSTEM, "system prompt")
        session.add_message(MessageRole.USER, "hi")
        session.add_message(MessageRole.ASSISTANT, "hello")
        msgs = session.to_anthropic_messages()
        assert len(msgs) == 2
        roles = [m["role"] for m in msgs]
        assert "system" not in roles
        assert msgs[0] == {"role": "user", "content": "hi"}
        assert msgs[1] == {"role": "assistant", "content": "hello"}

    def test_to_anthropic_messages_empty(self):
        session = self._make_session()
        assert session.to_anthropic_messages() == []

    def test_channel_type_values(self):
        assert ChannelType.FACEBOOK == "facebook"
        assert ChannelType.WHATSAPP == "whatsapp"
        assert ChannelType.API == "api"

    def test_message_role_values(self):
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
