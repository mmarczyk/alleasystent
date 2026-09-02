"""Unit tests for services/invoice_ledger.py — the duplicate-invoice guard.

Every assertion here is about one thing: an order must not be invoiced twice.
The ledger is what makes that decidable, since Allegro's own invoice list
only knows about invoices someone attached to the order.

REDIS_URL is deliberately unset, so these exercise the process-local fallback
store — the same code path, same semantics, no Redis needed in CI.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret")
    monkeypatch.delenv("REDIS_URL", raising=False)


@pytest.fixture(autouse=True)
def clear_ledger():
    from services import invoice_ledger
    invoice_ledger._MEMORY.clear()
    yield
    invoice_ledger._MEMORY.clear()


class TestClaim:
    async def test_first_claim_is_acquired(self):
        from services import invoice_ledger
        assert (await invoice_ledger.claim("user1", "ORD-1")).acquired is True

    async def test_second_claim_for_the_same_order_is_refused(self):
        """The whole point: two "wystaw fakturę dla ORD-1" in one day must
        result in one invoice, not two."""
        from services import invoice_ledger
        await invoice_ledger.claim("user1", "ORD-1")
        second = await invoice_ledger.claim("user1", "ORD-1")
        assert second.acquired is False
        assert second.record["status"] == invoice_ledger.STATUS_CLAIMED

    async def test_claims_are_per_order(self):
        from services import invoice_ledger
        await invoice_ledger.claim("user1", "ORD-1")
        assert (await invoice_ledger.claim("user1", "ORD-2")).acquired is True

    async def test_claims_are_per_user(self):
        from services import invoice_ledger
        await invoice_ledger.claim("user1", "ORD-1")
        assert (await invoice_ledger.claim("user2", "ORD-1")).acquired is True

    async def test_claim_after_issue_is_refused_and_reports_the_invoice(self):
        from services import invoice_ledger
        await invoice_ledger.claim("user1", "ORD-1")
        await invoice_ledger.mark_issued("user1", "ORD-1", "uuid-1", number="FV/1/2026")
        blocked = await invoice_ledger.claim("user1", "ORD-1")
        assert blocked.acquired is False
        assert blocked.record["invoice_uuid"] == "uuid-1"


class TestRelease:
    async def test_released_order_can_be_claimed_again(self):
        """A rejection by inFakt creates nothing, so the seller must be able
        to fix the data and try again."""
        from services import invoice_ledger
        await invoice_ledger.claim("user1", "ORD-1")
        await invoice_ledger.release("user1", "ORD-1")
        assert (await invoice_ledger.claim("user1", "ORD-1")).acquired is True

    async def test_release_of_an_unknown_order_is_harmless(self):
        from services import invoice_ledger
        await invoice_ledger.release("user1", "NEVER-SEEN")


class TestAlreadyHandled:
    async def test_reports_claimed_and_issued_orders(self):
        from services import invoice_ledger
        await invoice_ledger.claim("user1", "ORD-1")
        await invoice_ledger.claim("user1", "ORD-2")
        await invoice_ledger.mark_issued("user1", "ORD-2", "uuid-2")
        handled = await invoice_ledger.already_handled("user1", ["ORD-1", "ORD-2", "ORD-3"])
        assert handled == {"ORD-1", "ORD-2"}

    async def test_empty_input(self):
        from services import invoice_ledger
        assert await invoice_ledger.already_handled("user1", []) == set()


class TestDescribe:
    def test_issued_record_names_the_invoice_and_refuses_a_second(self):
        from services import invoice_ledger
        text = invoice_ledger.describe("ORD-1", {
            "status": invoice_ledger.STATUS_ISSUED,
            "invoice_uuid": "uuid-1",
            "number": "FV/7/2026",
            "issued_at": "2026-09-02T10:15:00",
        })
        assert "FV/7/2026" in text
        assert "uuid-1" in text
        assert "nie wystawiam drugiej" in text

    def test_unconfirmed_claim_points_at_the_infakt_panel(self):
        from services import invoice_ledger
        text = invoice_ledger.describe("ORD-1", {
            "status": invoice_ledger.STATUS_CLAIMED,
            "claimed_at": "2026-09-02T10:15:00",
        })
        assert "panelu inFakt" in text
        assert "duplikat" in text

    def test_missing_record_still_refuses(self):
        """A blocked claim whose record vanished is still a blocked claim —
        the message must never read like permission to issue."""
        from services import invoice_ledger
        text = invoice_ledger.describe("ORD-1", None)
        assert "nie zlecam tego drugi raz" in text
