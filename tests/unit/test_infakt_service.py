"""Unit tests for services/infakt_service.py — the async task polling.

inFakt creates invoices asynchronously, so "did it work?" is answered by
polling the task, not by the POST. Getting the pending/terminal split wrong
here shows up as an invoice the seller is told failed while inFakt goes on to
create it — hence the emphasis on the in-progress codes below.
"""
from __future__ import annotations

import httpx
import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret")
    monkeypatch.setenv("INFAKT_API_KEY", "test-infakt-key")
    # No Redis in unit tests: the duplicate ledger falls back to its
    # process-local store, which behaves identically for these assertions.
    monkeypatch.delenv("REDIS_URL", raising=False)


@pytest.fixture(autouse=True)
def clear_invoice_ledger():
    """The ledger is deliberately persistent — reset it between tests so one
    test's issuance doesn't block the next one's."""
    from services import invoice_ledger
    invoice_ledger._MEMORY.clear()
    yield
    invoice_ledger._MEMORY.clear()


TASK_REF = "task-ref-77123"
INVOICE_UUID = "af2b4e6c-0000-1111-2222-333344445555"


async def _service(status_payloads: list):
    """An InfaktService whose transport replays status_payloads, one per poll.

    An int entry stands for a status check that fails with that HTTP code.
    """
    from services.infakt_service import InfaktService

    remaining = list(status_payloads)
    calls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if request.method == "POST" and path.endswith("/async/invoices.json"):
            return httpx.Response(200, json={"invoice_task_reference_number": TASK_REF})
        if "/async/invoices/status/" in path:
            payload = remaining.pop(0) if remaining else status_payloads[-1]
            if isinstance(payload, int):  # bare HTTP status = a failing status check
                return httpx.Response(payload, json={"error": "boom"})
            return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": "Not Found"})

    svc = InfaktService()
    await svc._client.aclose()
    svc._client = httpx.AsyncClient(
        base_url="https://api.infakt.pl/api/v3",
        transport=httpx.MockTransport(handle),
    )
    return svc, calls


def _pending(code: int, description: str) -> dict:
    return {"processing_code": code, "processing_description": description}


_CREATED = {
    "processing_code": 201,
    "processing_description": "Faktura utworzona",
    "invoice_uuid": INVOICE_UUID,
}


class TestCreateInvoicePolling:
    async def test_success_on_first_check(self):
        svc, calls = await _service([_CREATED])
        status = await svc.create_invoice({"services": []}, poll_interval=0)
        assert status["invoice_uuid"] == INVOICE_UUID
        assert len([c for c in calls if "status" in c]) == 1
        await svc.aclose()

    async def test_code_140_is_still_processing_not_a_failure(self):
        """140 "Zlecenie jest w trakcie przetwarzania" used to be treated as a
        terminal failure, so an invoice caught mid-processing was reported to
        the seller as not issued — while inFakt created it seconds later."""
        svc, _ = await _service([
            _pending(140, "Zlecenie jest w trakcie przetwarzania"),
            _CREATED,
        ])
        status = await svc.create_invoice({"services": []}, poll_interval=0)
        assert status["invoice_uuid"] == INVOICE_UUID
        await svc.aclose()

    async def test_keeps_polling_through_the_whole_in_progress_range(self):
        svc, calls = await _service([
            _pending(100, "Zlecenie przyjęte"),
            _pending(140, "Zlecenie jest w trakcie przetwarzania"),
            _pending(150, "Zlecenie w kolejce"),
            _CREATED,
        ])
        status = await svc.create_invoice({"services": []}, poll_interval=0)
        assert status["invoice_uuid"] == INVOICE_UUID
        assert len([c for c in calls if "status" in c]) == 4
        await svc.aclose()

    async def test_unreadable_status_payload_is_not_a_failure(self):
        """A status without a usable processing_code says nothing about the
        task — keep checking rather than declaring the invoice failed."""
        svc, _ = await _service([{}, {"processing_code": None}, _CREATED])
        status = await svc.create_invoice({"services": []}, poll_interval=0)
        assert status["invoice_uuid"] == INVOICE_UUID
        await svc.aclose()

    async def test_real_rejection_still_raises(self):
        from services.infakt_service import InfaktTaskError

        svc, _ = await _service([{
            "processing_code": 400,
            "processing_description": "Nieprawidłowe dane faktury",
            "invoice_errors": ["client_tax_code is invalid"],
        }])
        with pytest.raises(InfaktTaskError) as exc_info:
            await svc.create_invoice({"services": []}, poll_interval=0)
        assert exc_info.value.processing_code == 400
        assert exc_info.value.errors == ["client_tax_code is invalid"]
        await svc.aclose()

    async def test_still_pending_after_max_attempts_times_out(self):
        svc, calls = await _service([_pending(140, "Zlecenie jest w trakcie przetwarzania")])
        with pytest.raises(TimeoutError) as exc_info:
            await svc.create_invoice({"services": []}, poll_interval=0, max_attempts=3)
        assert TASK_REF in str(exc_info.value)
        assert len([c for c in calls if "status" in c]) == 3
        await svc.aclose()

    async def test_transient_status_check_failure_is_retried(self):
        """A 5xx or a dropped connection on one poll says nothing about the
        task — the invoice is probably being created right then."""
        svc, _ = await _service([503, _CREATED])
        status = await svc.create_invoice({"services": []}, poll_interval=0)
        assert status["invoice_uuid"] == INVOICE_UUID
        await svc.aclose()

    async def test_task_reference_is_handed_over_before_any_polling(self):
        """The reference has to reach the caller even when the poll never
        resolves — that timed-out case is the only one that needs it."""
        seen: list[str] = []

        async def remember(task_ref: str) -> None:
            seen.append(task_ref)

        svc, _ = await _service([_pending(140, "Zlecenie jest w trakcie przetwarzania")])
        with pytest.raises(TimeoutError):
            await svc.create_invoice(
                {"services": []}, poll_interval=0, max_attempts=2, on_task_ref=remember,
            )
        assert seen == [TASK_REF]
        await svc.aclose()

    async def test_client_error_on_status_check_raises_immediately(self):
        """A 401/404 will not fix itself — don't spend the whole poll budget."""
        from services.infakt_service import InfaktAPIError

        svc, calls = await _service([401, _CREATED])
        with pytest.raises(InfaktAPIError):
            await svc.create_invoice({"services": []}, poll_interval=0)
        assert len([c for c in calls if "status" in c]) == 1
        await svc.aclose()


class TestIssueInvoiceForOrderMessages:
    async def test_timeout_warns_against_reissuing(self):
        """A task inFakt accepted but hasn't confirmed must not read like
        "nothing happened" — reissuing would create a duplicate invoice."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro = AsyncMock()
        allegro.get_order.return_value = type(
            "O", (), {"invoice_required": True, "buyer_login": "kupujacy"}
        )()
        allegro.get_order_invoices.return_value = []
        allegro.get_order_invoice_data.return_value = {"first_name": "Jan", "last_name": "Kowalski"}

        infakt = AsyncMock()
        infakt.create_invoice.side_effect = TimeoutError("inFakt invoice task task-ref-77123 still pending")

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert "NIE wystawiaj jej ponownie" in out
        assert "❌" not in out


def _fake_allegro(user_id: str = "user1", existing_invoices: list | None = None):
    """An AllegroService stand-in for one order that wants a VAT invoice and
    has nothing attached to it in Allegro — the state every duplicate started
    from."""
    from unittest.mock import AsyncMock

    allegro = AsyncMock()
    allegro._user_id = user_id
    allegro.get_order.return_value = type(
        "O", (), {"invoice_required": True, "buyer_login": "kupujacy"}
    )()
    allegro.get_order_invoices.return_value = existing_invoices or []
    allegro.get_order_invoice_data.return_value = {"first_name": "Jan", "last_name": "Kowalski"}
    return allegro


class TestIssueInvoiceForOrderDuplicateGuard:
    """Issuing the same order twice is the bug this guard exists for.

    Allegro reports an invoice only once its PDF has been ATTACHED to the
    order, which is a separate step — so "no invoice in Allegro" was true for
    an order already invoiced in inFakt, and the second ask issued a second
    real VAT invoice.
    """

    async def test_second_issuance_for_the_same_order_is_refused(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro = _fake_allegro()
        infakt = AsyncMock()
        infakt.create_invoice.return_value = {"invoice_uuid": INVOICE_UUID}
        infakt.get_share_link.return_value = "https://infakt.example/share/1"

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            first = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)
            second = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert "✅" in first
        assert infakt.create_invoice.await_count == 1  # NOT two invoices
        assert "nie wystawiam drugiej" in second
        assert INVOICE_UUID in second

    async def test_a_different_order_is_unaffected(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro = _fake_allegro()
        infakt = AsyncMock()
        infakt.create_invoice.return_value = {"invoice_uuid": INVOICE_UUID}
        infakt.get_share_link.return_value = "https://infakt.example/share/1"

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-2", is_production=False)

        assert "✅" in out
        assert infakt.create_invoice.await_count == 2

    async def test_rejection_by_infakt_leaves_the_order_issuable(self):
        """Nothing was created, so the seller must be able to fix the data and
        try again — the guard must not lock a never-issued order out."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service
        from services.infakt_service import InfaktTaskError

        allegro = _fake_allegro()
        infakt = AsyncMock()
        infakt.create_invoice.side_effect = [
            InfaktTaskError(400, "Nieprawidłowe dane faktury"),
            {"invoice_uuid": INVOICE_UUID},
        ]
        infakt.get_share_link.return_value = "https://infakt.example/share/1"

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            failed = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)
            retried = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert failed.startswith("❌")
        assert "✅" in retried

    async def test_a_timeout_blocks_the_next_attempt(self):
        """inFakt accepted the task, so the invoice probably exists — a second
        attempt would duplicate it, whatever the seller asks."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro = _fake_allegro()
        infakt = AsyncMock()
        infakt.create_invoice.side_effect = TimeoutError("still pending")

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)
            second = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert infakt.create_invoice.await_count == 1
        assert "duplikat" in second

    async def test_created_without_uuid_still_blocks_a_reissue(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro = _fake_allegro()
        infakt = AsyncMock()
        infakt.create_invoice.return_value = {"processing_code": 201}

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            first = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)
            second = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert "⚠️" in first
        assert infakt.create_invoice.await_count == 1
        assert "duplikat" in second

    async def test_failure_before_infakt_is_called_leaves_the_order_issuable(self):
        """Building the payload can fail on Allegro's side — inFakt never saw
        the order, so it must not stay claimed."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service
        from services import invoice_ledger

        allegro = _fake_allegro()
        allegro.get_order_invoice_data.side_effect = RuntimeError("Allegro down")
        infakt = AsyncMock()

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            with pytest.raises(RuntimeError):
                await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert await invoice_ledger.already_handled("user1", ["ORD-1"]) == set()

    async def test_invoice_already_attached_in_allegro_short_circuits(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro = _fake_allegro(existing_invoices=[{"id": "inv-1"}])
        infakt = AsyncMock()

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert "faktura już istnieje w Allegro" in out
        infakt.create_invoice.assert_not_awaited()


class TestIssueInvoiceForOrderShareLink:
    async def test_share_link_failure_does_not_read_as_a_failed_issuance(self):
        """The invoice exists; only the preview link couldn't be generated.
        Reporting that as "nie udało się wystawić" is what sent sellers back
        to issue the same invoice a second time."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service
        from services.infakt_service import InfaktAPIError

        allegro = _fake_allegro()
        infakt = AsyncMock()
        infakt.create_invoice.return_value = {"invoice_uuid": INVOICE_UUID}
        infakt.get_share_link.side_effect = InfaktAPIError(500, "boom")

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert "wystawiona w inFakt" in out
        assert INVOICE_UUID in out
        assert "nie wystawiaj jej ponownie" in out
        assert "Nie udało się wystawić faktury" not in out


class TestUnblockInvoiceForOrder:
    """The escape hatch for the guard above.

    The guard errs towards "the invoice probably exists", so an issuance that
    really did fail unrecorded would strand the order for good. Unblocking
    resolves that against inFakt — the seller's word only decides when there
    is nothing left to ask inFakt about.
    """

    async def _unblock(self, allegro, infakt, **kwargs):
        from unittest.mock import patch

        import services.infakt_service as infakt_service

        with patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            return await infakt_service.unblock_invoice_for_order(allegro, "ORD-1", **kwargs)

    async def _issue_with(self, allegro, infakt):
        """Run a real issuance so the ledger ends up in the state under test."""
        from unittest.mock import patch

        import services.infakt_service as infakt_service

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            return await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

    async def test_nothing_blocked_says_so(self):
        from unittest.mock import AsyncMock

        out = await self._unblock(_fake_allegro(), AsyncMock())
        assert "nic nie blokuje" in out

    async def test_invoice_missing_from_infakt_clears_the_block(self):
        """inFakt answering 404 for the recorded invoice is the one thing that
        proves the block was protecting nothing."""
        from unittest.mock import AsyncMock

        import services.infakt_service as infakt_service
        from services.infakt_service import InfaktAPIError

        allegro = _fake_allegro()
        infakt = AsyncMock()
        infakt.create_invoice.return_value = {"invoice_uuid": INVOICE_UUID}
        infakt.get_share_link.return_value = "https://infakt.example/share/1"
        await self._issue_with(allegro, infakt)

        infakt.get_invoice.side_effect = InfaktAPIError(404, "Not Found")
        out = await self._unblock(allegro, infakt)

        assert "✅" in out
        # …and the order is issuable again.
        infakt.create_invoice.reset_mock()
        assert "✅" in await self._issue_with(allegro, infakt)
        assert infakt.create_invoice.await_count == 1

    async def test_invoice_present_in_infakt_keeps_the_block(self):
        from unittest.mock import AsyncMock

        allegro = _fake_allegro()
        infakt = AsyncMock()
        infakt.create_invoice.return_value = {"invoice_uuid": INVOICE_UUID}
        infakt.get_share_link.return_value = "https://infakt.example/share/1"
        await self._issue_with(allegro, infakt)

        infakt.get_invoice.return_value = {"number": "FV/7/2026"}
        out = await self._unblock(allegro, infakt)

        assert "FV/7/2026" in out
        assert "nie zdejmuję blokady" in out

    async def test_infakt_unreachable_keeps_the_block(self):
        """Not knowing is not the same as knowing it isn't there."""
        from unittest.mock import AsyncMock

        from services.infakt_service import InfaktAPIError

        allegro = _fake_allegro()
        infakt = AsyncMock()
        infakt.create_invoice.return_value = {"invoice_uuid": INVOICE_UUID}
        infakt.get_share_link.return_value = "https://infakt.example/share/1"
        await self._issue_with(allegro, infakt)

        infakt.get_invoice.side_effect = InfaktAPIError(500, "boom")
        out = await self._unblock(allegro, infakt)

        assert "❌" in out
        assert "Nie zdejmuję blokady" in out

    async def test_attached_in_allegro_keeps_the_block(self):
        from unittest.mock import AsyncMock

        from services import invoice_ledger

        allegro = _fake_allegro(existing_invoices=[{"id": "inv-1"}])
        await invoice_ledger.mark_issued("user1", "ORD-1", INVOICE_UUID)

        out = await self._unblock(allegro, AsyncMock())
        assert "nie zdejmuję blokady" in out


class TestUnblockAfterATimeout:
    """A timed-out issuance leaves no invoice_uuid — only the inFakt task
    reference recorded before polling started. That reference is what makes
    the block recoverable without anyone guessing."""

    async def _timed_out_issuance(self, task_status_side_effect=None):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro = _fake_allegro()
        infakt = AsyncMock()

        async def create_invoice(payload, on_task_ref=None, **kwargs):
            await on_task_ref(TASK_REF)
            raise TimeoutError("still pending")

        infakt.create_invoice.side_effect = create_invoice

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)
        return allegro, infakt

    async def test_task_reference_is_recorded_before_the_polling_starts(self):
        from services import invoice_ledger

        await self._timed_out_issuance()
        record = await invoice_ledger.get("user1", "ORD-1")
        assert record["task_ref"] == TASK_REF

    async def test_a_failed_task_clears_the_block(self):
        from unittest.mock import patch

        import services.infakt_service as infakt_service

        allegro, infakt = await self._timed_out_issuance()
        infakt.get_task_status.return_value = {
            "processing_code": 400, "processing_description": "Nieprawidłowe dane faktury",
        }

        with patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            out = await infakt_service.unblock_invoice_for_order(allegro, "ORD-1")

        assert "✅" in out
        assert "Nieprawidłowe dane faktury" in out

    async def test_a_succeeded_task_keeps_the_block_and_recovers_the_uuid(self):
        """The invoice the timeout couldn't name — recovering its id here is
        what lets the seller attach it or send it to KSeF afterwards."""
        from unittest.mock import patch

        import services.infakt_service as infakt_service
        from services import invoice_ledger

        allegro, infakt = await self._timed_out_issuance()
        infakt.get_task_status.return_value = {
            "processing_code": 201, "invoice_uuid": INVOICE_UUID,
        }

        with patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            out = await infakt_service.unblock_invoice_for_order(allegro, "ORD-1")

        assert INVOICE_UUID in out
        assert "Nie zdejmuję blokady" in out
        record = await invoice_ledger.get("user1", "ORD-1")
        assert record["invoice_uuid"] == INVOICE_UUID

    async def test_a_still_processing_task_keeps_the_block(self):
        from unittest.mock import patch

        import services.infakt_service as infakt_service

        allegro, infakt = await self._timed_out_issuance()
        infakt.get_task_status.return_value = {
            "processing_code": 140, "processing_description": "Zlecenie jest w trakcie przetwarzania",
        }

        with patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            out = await infakt_service.unblock_invoice_for_order(allegro, "ORD-1")

        assert "nadal przetwarza" in out
        assert "✅" not in out


class TestUnblockOnTheSellersWord:
    """Last resort: no invoice id and no task reference to check against."""

    async def _claimed_only(self):
        from services import invoice_ledger

        allegro = _fake_allegro()
        await invoice_ledger.claim("user1", "ORD-1")
        return allegro

    async def test_without_explicit_confirmation_the_block_stays(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro = await self._claimed_only()
        with patch.object(infakt_service.InfaktService, "get_instance", return_value=AsyncMock()):
            out = await infakt_service.unblock_invoice_for_order(allegro, "ORD-1")

        assert "⚠️" in out
        assert "Sprawdź w panelu inFakt" in out

    async def test_explicit_confirmation_clears_the_block(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service
        from services import invoice_ledger

        allegro = await self._claimed_only()
        with patch.object(infakt_service.InfaktService, "get_instance", return_value=AsyncMock()):
            out = await infakt_service.unblock_invoice_for_order(
                allegro, "ORD-1", seller_confirmed=True,
            )

        assert "✅" in out
        assert await invoice_ledger.get("user1", "ORD-1") is None

    async def test_confirmation_does_not_override_a_verifiable_invoice(self):
        """seller_confirmed is a last resort, not an override — an invoice
        inFakt can still be asked about is checked, not taken on faith."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service
        from services import invoice_ledger

        allegro = _fake_allegro()
        await invoice_ledger.mark_issued("user1", "ORD-1", INVOICE_UUID)
        infakt = AsyncMock()
        infakt.get_invoice.return_value = {"number": "FV/7/2026"}

        with patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt):
            out = await infakt_service.unblock_invoice_for_order(
                allegro, "ORD-1", seller_confirmed=True,
            )

        assert "nie zdejmuję blokady" in out
        assert await invoice_ledger.get("user1", "ORD-1") is not None
