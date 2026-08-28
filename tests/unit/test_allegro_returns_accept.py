"""Unit tests for the Accept header used by the customer-returns endpoints.

/order/customer-returns is a beta resource: sending the default
public.v1 representation makes Allegro answer 406 NotAcceptableException.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.allegro_service import AllegroService

BETA = "application/vnd.allegro.beta.v1+json"


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


def _service(payload: dict) -> tuple[AllegroService, MagicMock]:
    service = AllegroService.__new__(AllegroService)
    service._get_headers = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "Authorization": "Bearer test",
            "Accept": "application/vnd.allegro.public.v1+json",
            "Content-Type": "application/vnd.allegro.public.v1+json",
        }
    )
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    service._client = client  # type: ignore[attr-defined]
    return service, client


class TestCustomerReturnsAccept:
    async def test_list_requests_beta_representation(self):
        service, client = _service({"customerReturns": [{"id": "r1"}]})

        returns = await service.get_customer_returns(limit=50, status="DELIVERED")

        assert returns == [{"id": "r1"}]
        _, kwargs = client.get.call_args
        assert kwargs["headers"]["Accept"] == BETA
        assert kwargs["params"] == {"limit": 50, "status": "DELIVERED"}

    async def test_detail_requests_beta_representation(self):
        service, client = _service({"id": "r1"})

        assert await service.get_customer_return("r1") == {"id": "r1"}

        args, kwargs = client.get.call_args
        assert args[0] == "/order/customer-returns/r1"
        assert kwargs["headers"]["Accept"] == BETA
