"""Unit tests for services/invoice_ledger.py — the record of invoices this
assistant has already issued, which is what stops the invoice reminder asking
for an invoice that exists but never reached Allegro."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")


def _mock_redis(**methods):
    r = AsyncMock()
    r.aclose = AsyncMock()
    for name, value in methods.items():
        setattr(r, name, AsyncMock(return_value=value))
    return r


class TestRecordIssued:
    @pytest.mark.asyncio
    async def test_writes_under_a_per_user_per_order_key(self):
        from services import invoice_ledger

        r = _mock_redis(set=True)
        with patch("redis.asyncio.from_url", return_value=r):
            await invoice_ledger.record_issued(
                "u1", "ord-1", invoice_uuid="inv-9", number="FV/1/2026", attached=True
            )
        key, raw = r.set.await_args[0]
        assert key == "allegro:invoice_issued:u1:ord-1"
        payload = json.loads(raw)
        assert payload["invoice_uuid"] == "inv-9"
        assert payload["number"] == "FV/1/2026"
        assert payload["attached"] is True

    @pytest.mark.asyncio
    async def test_records_a_failed_attachment_too(self):
        """The whole point: an invoice that exists in inFakt but not in Allegro
        must still be remembered, or the reminder asks for it again for ever."""
        from services import invoice_ledger

        r = _mock_redis(set=True)
        with patch("redis.asyncio.from_url", return_value=r):
            await invoice_ledger.record_issued(
                "u1", "ord-1", invoice_uuid="inv-9", attached=False, note="403"
            )
        payload = json.loads(r.set.await_args[0][1])
        assert payload["attached"] is False
        assert payload["note"] == "403"

    @pytest.mark.asyncio
    async def test_no_redis_is_a_silent_no_op(self, monkeypatch):
        from services import invoice_ledger

        monkeypatch.setenv("REDIS_URL", "")
        await invoice_ledger.record_issued("u1", "ord-1", invoice_uuid="inv-9")
        assert await invoice_ledger.get_records("u1", ["ord-1"]) == {}


class TestGetRecords:
    @pytest.mark.asyncio
    async def test_reads_a_batch_in_one_mget(self):
        from services import invoice_ledger

        r = _mock_redis(mget=[json.dumps({"invoice_uuid": "inv-9", "attached": False}), None])
        with patch("redis.asyncio.from_url", return_value=r):
            found = await invoice_ledger.get_records("u1", ["ord-1", "ord-2"])
        assert list(found) == ["ord-1"]
        assert r.mget.await_args[0][0] == [
            "allegro:invoice_issued:u1:ord-1",
            "allegro:invoice_issued:u1:ord-2",
        ]

    @pytest.mark.asyncio
    async def test_empty_input_never_touches_redis(self):
        from services import invoice_ledger

        r = _mock_redis(mget=[])
        with patch("redis.asyncio.from_url", return_value=r):
            assert await invoice_ledger.get_records("u1", []) == {}
        r.mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unreadable_entry_is_skipped_not_raised(self):
        from services import invoice_ledger

        r = _mock_redis(mget=["{not json"])
        with patch("redis.asyncio.from_url", return_value=r):
            assert await invoice_ledger.get_records("u1", ["ord-1"]) == {}


class TestUserIdOf:
    def test_reads_the_services_user(self):
        from services import invoice_ledger

        class _Svc:
            _user_id = "seller-7"

        assert invoice_ledger.user_id_of(_Svc()) == "seller-7"

    def test_falls_back_to_default(self):
        from services import invoice_ledger

        assert invoice_ledger.user_id_of(object()) == "default"
