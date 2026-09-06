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


class TestIssuingAlsoAttaches:
    """Issuing used to stop at inFakt, so Allegro still reported the order as
    having no invoice and the reminder (services/invoice_reminder.py), which
    asks Allegro exactly that, nagged about the same order every two hours for
    ever — its own "wystaw" could never satisfy its own condition."""

    @staticmethod
    def _allegro():
        from unittest.mock import AsyncMock

        allegro = AsyncMock()
        allegro._user_id = "u1"
        allegro.get_order.return_value = type(
            "O", (), {"invoice_required": True, "buyer_login": "kupujacy"}
        )()
        allegro.get_order_invoices.return_value = []
        allegro.get_order_invoice_data.return_value = {"company_name": "Firma sp. z o.o."}
        allegro.create_order_invoice_record.return_value = "allegro-inv-1"
        return allegro

    @staticmethod
    def _infakt():
        from unittest.mock import AsyncMock

        infakt = AsyncMock()
        infakt.create_invoice.return_value = {"invoice_uuid": "inv-9"}
        infakt.get_share_link.return_value = "https://app.infakt.pl/share/inv-9"
        infakt.get_invoice.return_value = {"number": "FV/1/2026"}
        infakt.get_invoice_pdf.return_value = b"%PDF-1.4 tiny"
        return infakt

    async def test_invoice_is_attached_to_the_allegro_order(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro, infakt = self._allegro(), self._infakt()
        record = AsyncMock()
        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt), \
             patch("services.invoice_ledger.record_issued", record):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        allegro.create_order_invoice_record.assert_awaited_once_with("ORD-1", "FV/1/2026", "faktura-FV/1/2026.pdf")
        allegro.upload_order_invoice_file.assert_awaited_once()
        assert "Dołączona do zamówienia w Allegro" in out
        assert record.await_args.kwargs["attached"] is True

    async def test_failed_attachment_is_reported_and_recorded_not_swallowed(self):
        """The invoice exists — saying "not issued" next time would push the
        seller into issuing a second, irreversible one."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service
        from services.allegro_service import AllegroAPIError

        allegro, infakt = self._allegro(), self._infakt()
        allegro.create_order_invoice_record.side_effect = AllegroAPIError(403, "Forbidden")
        record = AsyncMock()
        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt), \
             patch("services.invoice_ledger.record_issued", record):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert "Nie udało się dołączyć" in out
        assert "nie wystawiaj jej ponownie" in out.lower()
        assert record.await_args.kwargs["attached"] is False
        assert record.await_args.kwargs["invoice_uuid"] == "inv-9"

    async def test_oversized_pdf_is_caught_before_the_record_is_created(self):
        """The POST would otherwise leave an empty invoice record on the order."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service
        from services.allegro_service import INVOICE_FILE_MAX_BYTES

        allegro, infakt = self._allegro(), self._infakt()
        infakt.get_invoice_pdf.return_value = b"x" * (INVOICE_FILE_MAX_BYTES + 1)
        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt), \
             patch("services.invoice_ledger.record_issued", AsyncMock()):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        allegro.create_order_invoice_record.assert_not_awaited()
        assert "Nie udało się dołączyć" in out

    async def test_a_timed_out_issuance_is_recorded_so_it_is_not_nagged_again(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro, infakt = self._allegro(), self._infakt()
        infakt.create_invoice.side_effect = TimeoutError("still pending")
        record = AsyncMock()
        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt), \
             patch("services.invoice_ledger.record_issued", record):
            await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        record.assert_awaited_once()
        assert record.await_args.kwargs["attached"] is False


class TestNeverIssuesTwiceForOneOrder:
    """Allegro reporting "no invoice" is the truth about the ORDER, not about
    inFakt: when the attachment failed (a token without orders:write comes back
    403), a real numbered invoice exists while Allegro correctly still says the
    order has none. Issuing again would create a second one, which cannot be
    undone — so the earlier one gets finished instead."""

    @staticmethod
    def _allegro(has_scope=True):
        from unittest.mock import AsyncMock, MagicMock

        allegro = AsyncMock()
        allegro._user_id = "u1"
        allegro.has_scope = MagicMock(return_value=has_scope)
        allegro.get_order.return_value = type(
            "O", (), {"invoice_required": True, "buyer_login": "kupujacy"}
        )()
        allegro.get_order_invoices.return_value = []
        allegro.get_order_invoice_data.return_value = {"company_name": "Firma"}
        allegro.create_order_invoice_record.return_value = "allegro-inv-1"
        return allegro

    @staticmethod
    def _infakt():
        from unittest.mock import AsyncMock

        infakt = AsyncMock()
        infakt.create_invoice.return_value = {"invoice_uuid": "inv-NEW"}
        infakt.get_share_link.return_value = "https://app.infakt.pl/share/inv-NEW"
        infakt.get_invoice.return_value = {"number": "FV/1/2026"}
        infakt.get_invoice_pdf.return_value = b"%PDF-1.4 tiny"
        return infakt

    async def test_a_known_issuance_is_attached_not_reissued(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro, infakt = self._allegro(), self._infakt()
        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt), \
             patch("services.invoice_ledger.get_record",
                   AsyncMock(return_value={"invoice_uuid": "inv-OLD", "number": "FV/9/2026"})), \
             patch("services.invoice_ledger.record_issued", AsyncMock()):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        infakt.create_invoice.assert_not_awaited()
        infakt.get_invoice_pdf.assert_awaited_once_with("inv-OLD")
        allegro.upload_order_invoice_file.assert_awaited_once()
        assert "nie wystawiałem drugiej" in out

    async def test_a_still_failing_reattach_does_not_reissue_either(self):
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service
        from services.allegro_service import AllegroAPIError

        allegro, infakt = self._allegro(has_scope=False), self._infakt()
        allegro.create_order_invoice_record.side_effect = AllegroAPIError(403, "Forbidden")
        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt), \
             patch("services.invoice_ledger.get_record",
                   AsyncMock(return_value={"invoice_uuid": "inv-OLD", "number": "FV/9/2026"})), \
             patch("services.invoice_ledger.record_issued", AsyncMock()):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        infakt.create_invoice.assert_not_awaited()
        assert "nie wystawiam drugiej" in out

    async def test_a_403_names_the_missing_permission(self):
        """The seller otherwise has to guess why it failed before going and
        attaching the PDF by hand."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service
        from services.allegro_service import AllegroAPIError

        allegro, infakt = self._allegro(has_scope=False), self._infakt()
        allegro.create_order_invoice_record.side_effect = AllegroAPIError(403, "Forbidden")
        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt), \
             patch("services.invoice_ledger.get_record", AsyncMock(return_value=None)), \
             patch("services.invoice_ledger.record_issued", AsyncMock()):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        assert "allegro:api:orders:write" in out
        assert "apps.developer.allegro.pl" in out
        assert "nie wystawiaj jej ponownie" in out.lower()
