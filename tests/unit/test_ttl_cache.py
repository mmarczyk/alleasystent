"""Unit tests for _TTLCache, AllegroAuthError, AllegroAPIError, and _parse_order."""
from __future__ import annotations

from time import monotonic
from unittest.mock import MagicMock, patch

import pytest


class TestTTLCache:
    def _cache(self, ttl=60.0):
        from services.allegro_service import _TTLCache
        return _TTLCache(ttl=ttl)

    def test_get_returns_none_for_missing_key(self):
        c = self._cache()
        assert c.get("missing") is None

    def test_set_and_get(self):
        c = self._cache()
        c.set("k", "value")
        assert c.get("k") == "value"

    def test_get_returns_none_after_ttl(self):
        c = self._cache(ttl=1.0)
        c.set("k", "v")
        future_time = monotonic() + 2.0
        with patch("services.allegro_service.monotonic", return_value=future_time):
            assert c.get("k") is None

    def test_expired_key_is_removed_on_get(self):
        c = self._cache(ttl=1.0)
        c.set("k", "v")
        future_time = monotonic() + 2.0
        with patch("services.allegro_service.monotonic", return_value=future_time):
            c.get("k")
        assert "k" not in c._store

    def test_invalidate_removes_key(self):
        c = self._cache()
        c.set("k", "v")
        c.invalidate("k")
        assert c.get("k") is None

    def test_invalidate_missing_key_no_error(self):
        c = self._cache()
        c.invalidate("nonexistent")  # should not raise

    def test_clear_removes_all(self):
        c = self._cache()
        c.set("a", 1)
        c.set("b", 2)
        c.clear()
        assert c.get("a") is None
        assert c.get("b") is None

    def test_set_overwrites_existing(self):
        c = self._cache()
        c.set("k", "old")
        c.set("k", "new")
        assert c.get("k") == "new"

    def test_stores_none_value(self):
        c = self._cache()
        c.set("k", None)
        # None stored — but get() returns None for missing too
        # The value is in the store
        assert "k" in c._store


class TestAllegroExceptions:
    def test_allegro_auth_error_is_exception(self):
        from services.allegro_service import AllegroAuthError
        err = AllegroAuthError("not authenticated")
        assert isinstance(err, Exception)
        assert str(err) == "not authenticated"

    def test_allegro_api_error_has_status_code(self):
        from services.allegro_service import AllegroAPIError
        err = AllegroAPIError(404, "not found")
        assert err.status_code == 404
        assert "404" in str(err)
        assert "not found" in str(err)

    def test_allegro_api_error_is_exception(self):
        from services.allegro_service import AllegroAPIError
        err = AllegroAPIError(500, "server error")
        assert isinstance(err, Exception)


class TestAllegroServiceParsing:
    def _make_service(self, monkeypatch):
        """Create an AllegroService with minimal env and mocked I/O."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test")
        # Patch file system operations so __init__ doesn't read real files
        with patch("services.allegro_service.Path.exists", return_value=False), \
             patch("services.allegro_service.Path.read_text", return_value=""):
            from services.allegro_service import AllegroService
            svc = AllegroService.__new__(AllegroService)
            svc._user_id = "default"
            svc._settings = MagicMock()
            svc._settings.allegro_token_file = ".allegro_tokens.json"
            svc._settings.allegro_mock_token = ""
            svc._settings.redis_url = ""
            svc._tokens = None
            svc._pending_device_code = None
            svc._redis = None
            from services.allegro_service import _TTLCache
            svc._order_cache = _TTLCache(ttl=300.0)
            svc._orders_list_cache = _TTLCache(ttl=60.0)
            svc._invoice_cache = _TTLCache(ttl=120.0)
            svc._all_offers_cache = _TTLCache(ttl=300.0)
            return svc

    def test_parse_order_basic(self, monkeypatch):
        svc = self._make_service(monkeypatch)
        data = {
            "id": "order-123",
            "buyer": {"login": "buyer1", "email": "buyer@example.com"},
            "status": "READY_FOR_PROCESSING",
            "fulfillment": {"status": "NEW"},
            "payment": {"type": "ONLINE", "finishedAt": "2024-01-01T10:00:00Z"},
            "summary": {"totalToPay": {"amount": "150.00", "currency": "PLN"}},
            "lineItems": [
                {
                    "offer": {"id": "off-1", "name": "Widget"},
                    "quantity": 2,
                    "price": {"amount": "75.00", "currency": "PLN"},
                }
            ],
            "delivery": {"method": {"name": "InPost"}},
            "invoice": {"required": False},
            "boughtAt": "2024-01-01T09:00:00Z",
        }
        order = svc._parse_order(data)
        assert order.order_id == "order-123"
        assert order.buyer_login == "buyer1"
        assert order.buyer_email == "buyer@example.com"
        assert order.status == "READY_FOR_PROCESSING"
        assert order.total_price == pytest.approx(150.0)
        assert len(order.line_items) == 1
        assert order.line_items[0].offer_name == "Widget"
        assert order.line_items[0].quantity == 2

    def test_parse_order_empty_delivery(self, monkeypatch):
        svc = self._make_service(monkeypatch)
        data = {
            "id": "o2",
            "buyer": {"login": "b"},
            "status": "BOUGHT",
            "delivery": None,
        }
        order = svc._parse_order(data)
        assert order.delivery == {}

    def test_parse_order_no_line_items(self, monkeypatch):
        svc = self._make_service(monkeypatch)
        data = {"id": "o3", "buyer": {"login": "b"}, "status": "BOUGHT"}
        order = svc._parse_order(data)
        assert order.line_items == []

    def test_token_file_default_user(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test")
        with patch("services.allegro_service.Path.exists", return_value=False):
            from services.allegro_service import AllegroService
            svc = AllegroService.__new__(AllegroService)
            svc._user_id = "default"
            svc._settings = MagicMock()
            svc._settings.allegro_token_file = ".allegro_tokens.json"
            from pathlib import Path
            svc._token_file = AllegroService._token_file.__get__(svc)
            path = svc._token_file()
            assert str(path) == ".allegro_tokens.json"

    def test_token_file_non_default_user(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test")
        with patch("services.allegro_service.Path.exists", return_value=False):
            from services.allegro_service import AllegroService
            svc = AllegroService.__new__(AllegroService)
            svc._user_id = "seller123"
            svc._settings = MagicMock()
            svc._settings.allegro_token_file = ".allegro_tokens.json"
            svc._token_file = AllegroService._token_file.__get__(svc)
            path = svc._token_file()
            assert "seller123" in str(path)

    def test_redis_tokens_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test")
        with patch("services.allegro_service.Path.exists", return_value=False):
            from services.allegro_service import AllegroService
            svc = AllegroService.__new__(AllegroService)
            svc._user_id = "myuser"
            assert AllegroService._redis_tokens_key.fget(svc) == "allegro:tokens:myuser"
