"""Unit tests for token-scope diagnosis behind the "brak uprawnień" errors.

Allegro gives a token exactly the scopes the app was authorized with and
answers 403 for anything else, so both failures the seller hit — costs in the
sales summary and attaching an invoice to an order — surface as the same
useless "no permission". These tests pin that the token is asked what it can
actually do, and that the answer decides what the seller is told.
"""
from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.allegro.allegro_agent import AllegroAgent
from services.allegro_service import (
    SCOPE_BILLING_READ,
    SCOPE_ORDERS_WRITE,
    AllegroAPIError,
    AllegroService,
    decode_token_scopes,
)


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


def _jwt(payload: dict) -> str:
    """An unsigned JWT — decode_token_scopes reads the payload, not the signature."""
    def seg(data: dict) -> str:
        raw = json.dumps(data).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig"


class TestDecodeTokenScopes:
    def test_reads_a_space_separated_scope_claim(self):
        token = _jwt({"scope": "allegro:api:orders:read allegro:api:orders:write"})

        assert decode_token_scopes(token) == [
            "allegro:api:orders:read", "allegro:api:orders:write",
        ]

    def test_reads_a_list_scope_claim(self):
        token = _jwt({"scope": ["allegro:api:billing:read"]})

        assert decode_token_scopes(token) == ["allegro:api:billing:read"]

    def test_an_unreadable_token_is_unknown_not_empty(self):
        """A mock/opaque token must not be reported as "you have no permissions"."""
        assert decode_token_scopes("not-a-jwt") is None
        assert decode_token_scopes("") is None
        assert decode_token_scopes(_jwt({"sub": "x"})) is None


class TestServiceScopeChecks:
    def _service(self, token: str | None):
        service = AllegroService.__new__(AllegroService)
        if token is None:
            service._tokens = None  # type: ignore[attr-defined]
        else:
            service._tokens = MagicMock(access_token=token)  # type: ignore[attr-defined]
        return service

    def test_has_scope_is_true_when_the_token_carries_it(self):
        service = self._service(_jwt({"scope": f"{SCOPE_ORDERS_WRITE} {SCOPE_BILLING_READ}"}))

        assert service.has_scope(SCOPE_ORDERS_WRITE) is True

    def test_has_scope_is_false_when_the_token_lacks_it(self):
        service = self._service(_jwt({"scope": "allegro:api:orders:read"}))

        assert service.has_scope(SCOPE_ORDERS_WRITE) is False

    def test_has_scope_is_none_when_it_cannot_be_read(self):
        assert self._service("opaque-mock-token").has_scope(SCOPE_ORDERS_WRITE) is None
        assert self._service(None).has_scope(SCOPE_ORDERS_WRITE) is None


class TestErrorBodyParsing:
    """Allegro explains its refusals in the response body; the raw JSON used to
    be pasted at the seller, burying the one sentence that says what to do."""

    def _response(self, status: int, body, is_json: bool = True):
        response = MagicMock(status_code=status)
        if is_json:
            response.json.return_value = body
            response.text = json.dumps(body)
        else:
            response.json.side_effect = ValueError("not json")
            response.text = body
        return response

    def test_the_user_message_becomes_the_error_detail(self):
        from services.allegro_service import _api_error

        exc = _api_error(self._response(403, {"errors": [{
            "code": "AccessDenied",
            "message": "Access is denied",
            "userMessage": "Nie masz uprawnień do tej operacji",
        }]}))

        assert exc.status_code == 403
        assert exc.code == "AccessDenied"
        assert exc.user_message == "Nie masz uprawnień do tej operacji"
        assert "Nie masz uprawnień do tej operacji" in str(exc)

    def test_a_non_json_body_still_reaches_the_caller(self):
        from services.allegro_service import _api_error

        exc = _api_error(self._response(502, "<html>Bad Gateway</html>", is_json=False))

        assert exc.status_code == 502
        assert "Bad Gateway" in str(exc)


def _agent(token_scopes: list[str] | None = None):
    with patch("agents.base_agent.AsyncOpenAI"), \
         patch("agents.allegro.allegro_agent.AllegroService") as MockService:
        service = MagicMock()
        service._tokens = MagicMock()
        service._tokens.is_expired.return_value = False
        service.token_scopes.return_value = token_scopes
        service.has_scope.side_effect = (
            lambda scope: None if token_scopes is None else scope in token_scopes
        )
        MockService.get_instance.return_value = service
        return AllegroAgent()


def _attach(agent, pdf: bytes = b"%PDF-1.4 short"):
    """Wire inFakt so only the Allegro half of the attach can fail."""
    infakt = MagicMock()
    infakt.get_invoice = AsyncMock(return_value={"number": "FV/1/2026"})
    infakt.get_invoice_pdf = AsyncMock(return_value=pdf)
    return patch("services.infakt_service.InfaktService.get_instance", return_value=infakt)


class TestInvoiceAttachPermissionError:
    async def test_a_missing_scope_is_named_with_the_fix(self):
        agent = _agent(token_scopes=["allegro:api:orders:read"])
        agent._allegro.create_order_invoice_record = AsyncMock(
            side_effect=AllegroAPIError(403, "Brak uprawnień", code="AccessDenied")
        )

        with _attach(agent):
            result = await agent._dispatch(
                "attach_invoice_to_allegro_order", {"order_id": "o1", "invoice_uuid": "u1"}
            )

        assert SCOPE_ORDERS_WRITE in result
        assert "apps.developer.allegro.pl" in result
        assert "/allegro/login" in result

    async def test_a_403_despite_the_scope_says_the_token_is_not_the_problem(self):
        agent = _agent(token_scopes=[SCOPE_ORDERS_WRITE])
        agent._allegro.create_order_invoice_record = AsyncMock(
            side_effect=AllegroAPIError(
                403, "Nie masz dostępu do tego zamówienia", code="AccessDenied",
                user_message="Nie masz dostępu do tego zamówienia",
            )
        )

        with _attach(agent):
            result = await agent._dispatch(
                "attach_invoice_to_allegro_order", {"order_id": "o1", "invoice_uuid": "u1"}
            )

        assert "konta" in result
        assert "apps.developer.allegro.pl" not in result
        assert "Nie masz dostępu do tego zamówienia" in result

    async def test_an_oversized_pdf_is_rejected_before_the_invoice_record_is_created(self):
        """The record and the file are two calls — failing on the second would
        leave an empty invoice on the order for a retry to trip over."""
        agent = _agent(token_scopes=[SCOPE_ORDERS_WRITE])
        agent._allegro.create_order_invoice_record = AsyncMock()

        with _attach(agent, pdf=b"x" * (4 * 1024 * 1024)):
            result = await agent._dispatch(
                "attach_invoice_to_allegro_order", {"order_id": "o1", "invoice_uuid": "u1"}
            )

        assert "MB" in result
        agent._allegro.create_order_invoice_record.assert_not_awaited()


class TestBillingScopeMessage:
    def _failing_agent(self, token_scopes):
        from models.allegro import AllegroOrder, AllegroOrderLine

        agent = _agent(token_scopes=token_scopes)
        agent._allegro.get_all_paid_orders_in_period = AsyncMock(return_value=[
            AllegroOrder(
                order_id="a", buyer_login="jan", status="READY_FOR_PROCESSING",
                fulfillment_status="SENT", total_price=100.0, currency="PLN",
                created_at="2026-01-10T10:00:00Z", paid_at="2026-01-10T10:00:00Z",
                line_items=[AllegroOrderLine(offer_id="1", offer_name="Włóczka", quantity=1, price=100.0)],
            )
        ])
        agent._allegro.get_billing_entries_in_period = AsyncMock(
            side_effect=AllegroAPIError(403, "Brak uprawnień", code="AccessDenied")
        )
        return agent

    async def test_the_missing_billing_scope_is_named(self):
        agent = self._failing_agent(["allegro:api:orders:read"])

        result = await agent._dispatch(
            "get_sales_summary", {"date_from_local": "2026-01-01", "date_to_local": "2026-12-31"}
        )

        assert SCOPE_BILLING_READ in result
        assert "apps.developer.allegro.pl" in result

    async def test_an_unreadable_token_does_not_accuse_the_seller_of_missing_it(self):
        agent = self._failing_agent(None)

        result = await agent._dispatch(
            "get_sales_summary", {"date_from_local": "2026-01-01", "date_to_local": "2026-12-31"}
        )

        assert "nie ma uprawnienia" not in result
        assert SCOPE_BILLING_READ in result


class TestAccountInfoScopes:
    async def test_account_info_lists_the_scopes_and_flags_the_missing_ones(self):
        agent = _agent(token_scopes=["allegro:api:orders:read"])
        agent._allegro.get_user_info = AsyncMock(return_value={
            "login": "sklep", "email": "a@b.pl", "registeredAt": "2020-01-01T10:00:00Z",
        })

        result = await agent._dispatch("get_account_info", {})

        assert "allegro:api:orders:read" in result
        assert SCOPE_BILLING_READ in result and SCOPE_ORDERS_WRITE in result
        assert "403" in result

    async def test_no_scope_section_when_the_token_cannot_be_read(self):
        agent = _agent(token_scopes=None)
        agent._allegro.get_user_info = AsyncMock(return_value={"login": "sklep"})

        result = await agent._dispatch("get_account_info", {})

        assert "scope" not in result
        assert "Login: **sklep**" in result
