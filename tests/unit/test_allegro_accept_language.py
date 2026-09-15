"""Unit test for the Accept-Language header every Allegro call carries.

Allegro translates the labels it sends back according to this header and falls
back to English without it. That is not cosmetic: the billing type name is the
only signal saying a fee is the shipment, and the rules that read it are
written in Polish — so a missing header turned "Opłata za dostawę ORLEN Paczka
Allegro Delivery" into an unrecognised English label.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from models.allegro import AllegroTokens
from services.allegro_service import AllegroService


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


class TestAcceptLanguage:
    async def test_headers_ask_allegro_for_polish_labels(self):
        service = AllegroService.__new__(AllegroService)
        service._tokens = MagicMock(spec=AllegroTokens)
        service._tokens.access_token = "test"
        service._tokens.is_expired.return_value = False

        headers = await service._get_headers()

        assert headers["Accept-Language"] == "pl-PL"

    async def test_billing_fetch_sends_that_header(self):
        service = AllegroService.__new__(AllegroService)
        service._get_headers = AsyncMock(  # type: ignore[method-assign]
            return_value={"Authorization": "Bearer test", "Accept-Language": "pl-PL"}
        )
        response = MagicMock(status_code=200)
        response.json.return_value = {"billingEntries": []}
        client = MagicMock()
        client.get = AsyncMock(return_value=response)
        service._client = client  # type: ignore[attr-defined]

        await service.get_billing_entries_for_order("abc-123")

        assert client.get.call_args.kwargs["headers"]["Accept-Language"] == "pl-PL"
