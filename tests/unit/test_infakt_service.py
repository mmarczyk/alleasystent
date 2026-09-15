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


class TestKsefIsForBusinessInvoicesOnly:
    """KSeF carries invoices between BUSINESSES and addresses the buyer by NIP.
    An invoice for a buyer without one has no place there, so filing it is a
    mistake that cannot be withdrawn.

    Who the buyer is comes from ALLEGRO — the buyer declares the company and the
    NIP when they order. What is in inFakt is only the copy we wrote there from
    that same declaration, so checking it would be checking our own homework."""

    @staticmethod
    def _allegro(address: dict):
        from unittest.mock import AsyncMock

        allegro = AsyncMock()
        allegro.get_order_invoice_data.return_value = address
        return allegro

    @staticmethod
    async def _service():
        """An InfaktService that records every request, so a POST that should
        never happen is visible."""
        from services.infakt_service import InfaktService

        calls: list[tuple[str, str]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            calls.append((request.method, request.url.path))
            return httpx.Response(200, json={"status": "sent"})

        svc = InfaktService()
        await svc._client.aclose()
        svc._client = httpx.AsyncClient(
            base_url="https://api.infakt.pl/api/v3",
            transport=httpx.MockTransport(handle),
        )
        return svc, calls

    def test_a_company_with_a_nip_may_be_filed(self):
        from services.infakt_service import ksef_refusal_reason

        assert ksef_refusal_reason({
            "company_name": "Firma sp. z o.o.", "vat_id": "5252445767",
        }) is None

    def test_a_private_person_may_not(self):
        from services.infakt_service import ksef_refusal_reason

        reason = ksef_refusal_reason({"first_name": "Anna", "last_name": "Kowalska"})
        assert reason == "nabywcą jest osoba prywatna"

    def test_a_company_without_a_nip_may_not_either(self):
        """Kept apart from the private person on purpose: this one the seller
        can go and look at on the order."""
        from services.infakt_service import ksef_refusal_reason

        reason = ksef_refusal_reason({"company_name": "Firma bez NIP", "vat_id": ""})
        assert reason and "nie ma NIP-u" in reason

    def test_an_empty_address_is_refused_not_waved_through(self):
        from services.infakt_service import ksef_refusal_reason

        assert ksef_refusal_reason({}) is not None

    async def test_the_request_is_refused_before_it_is_sent(self):
        from services.infakt_service import KsefNotAllowedError

        svc, calls = await self._service()
        allegro = self._allegro({"first_name": "Anna", "last_name": "Kowalska"})
        with pytest.raises(KsefNotAllowedError) as exc_info:
            await svc.send_to_ksef("inv-1", allegro=allegro, order_id="ORD-1")
        await svc.aclose()

        assert not calls, calls
        assert exc_info.value.reason == "nabywcą jest osoba prywatna"

    async def test_allegro_is_what_gets_asked(self):
        """Not inFakt: the invoice in inFakt says whatever we put in it, and for
        an order that never had a NIP that would be the wrong answer for ever."""
        svc, _ = await self._service()
        allegro = self._allegro({"company_name": "Firma", "vat_id": "5252445767"})
        await svc.send_to_ksef("inv-2", allegro=allegro, order_id="ORD-2")
        await svc.aclose()

        allegro.get_order_invoice_data.assert_awaited_once_with("ORD-2")

    async def test_a_company_invoice_still_goes_through(self):
        svc, calls = await self._service()
        allegro = self._allegro({"company_name": "Firma", "vat_id": "5252445767"})
        result = await svc.send_to_ksef("inv-2", allegro=allegro, order_id="ORD-2")
        await svc.aclose()

        assert result == {"status": "sent"}
        assert [c for c in calls if c[0] == "POST"]

    def test_the_check_cannot_be_skipped_by_forgetting_it(self):
        """allegro/order_id are keyword-only and have no defaults, so a caller
        that does not make the check possible does not get to send at all."""
        import inspect

        from services.infakt_service import InfaktService

        params = inspect.signature(InfaktService.send_to_ksef).parameters
        for name in ("allegro", "order_id"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert params[name].default is inspect.Parameter.empty


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


class TestIssuingStopsBeforeTheBuyerSeesIt:
    """Issuing and attaching were briefly one step, to stop the reminder
    (services/invoice_reminder.py) asking about an order Allegro calls
    uninvoiced until a PDF is on it. That put the invoice on the BUYER's order
    page before anyone had checked it — and Allegro takes one invoice per
    order, so a wrong NIP could not be taken back. The attachment is the
    seller's call again; the ledger is what keeps the reminder honest."""

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

    async def _issue(self, allegro, infakt, record):
        from unittest.mock import patch

        import services.infakt_service as infakt_service

        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt), \
             patch("services.invoice_ledger.record_issued", record):
            return await infakt_service.issue_invoice_for_order(
                allegro, "ORD-1", is_production=False
            )

    async def test_nothing_is_uploaded_to_allegro(self):
        from unittest.mock import AsyncMock

        allegro, infakt = self._allegro(), self._infakt()
        out = await self._issue(allegro, infakt, AsyncMock())

        allegro.create_order_invoice_record.assert_not_awaited()
        allegro.upload_order_invoice_file.assert_not_awaited()
        infakt.get_invoice_pdf.assert_not_awaited()
        assert "NIE dołączyłem" in out

    async def test_the_seller_is_told_how_to_confirm_it(self):
        """An issued invoice nobody can act on is worse than no invoice: the
        seller has to be able to see it and know the words that attach it."""
        from unittest.mock import AsyncMock

        out = await self._issue(self._allegro(), self._infakt(), AsyncMock())

        assert "https://app.infakt.pl/share/inv-9" in out
        assert "dołącz fakturę do zamówienia `ORD-1`" in out
        assert "inv-9" in out

    async def test_the_issuance_is_recorded_as_not_attached(self):
        """Allegro will keep calling the order uninvoiced — correctly. The
        record is what stops the next "wystaw" creating a second invoice."""
        from unittest.mock import AsyncMock

        record = AsyncMock()
        await self._issue(self._allegro(), self._infakt(), record)

        assert record.await_args.kwargs["attached"] is False
        assert record.await_args.kwargs["invoice_uuid"] == "inv-9"

    async def test_a_private_person_invoice_says_ksef_is_not_an_option(self):
        """Said at issuance so the seller does not ask for something that will
        be refused two messages later."""
        from unittest.mock import AsyncMock

        allegro = self._allegro()
        allegro.get_order_invoice_data.return_value = {"first_name": "Anna", "last_name": "Kowalska"}
        out = await self._issue(allegro, self._infakt(), AsyncMock())

        assert "osoba prywatna" in out
        assert "Do KSeF ta faktura NIE pójdzie" in out

    async def test_a_company_invoice_still_offers_ksef_on_request(self):
        from unittest.mock import AsyncMock

        allegro = self._allegro()
        allegro.get_order_invoice_data.return_value = {
            "company_name": "Firma sp. z o.o.", "vat_id": "5252445767",
        }
        out = await self._issue(allegro, self._infakt(), AsyncMock())

        assert "Nabywca: firma" in out
        assert "Do KSeF też wysyłam wyłącznie na Twoje wyraźne polecenie" in out

    async def test_a_company_without_a_nip_on_the_order_is_flagged_at_issuance(self):
        """The invoice is fine, KSeF is not — and the seller can still fix the
        NIP on the order before asking for it."""
        from unittest.mock import AsyncMock

        allegro = self._allegro()
        allegro.get_order_invoice_data.return_value = {"company_name": "Firma bez NIP"}
        out = await self._issue(allegro, self._infakt(), AsyncMock())

        assert "Do KSeF ta faktura NIE pójdzie" in out
        assert "nie ma NIP-u" in out

    async def test_a_missing_share_link_does_not_fail_the_issuance(self):
        from unittest.mock import AsyncMock

        import services.infakt_service as infakt_service

        infakt = self._infakt()
        infakt.get_share_link.side_effect = infakt_service.InfaktAPIError(500, "boom")
        out = await self._issue(self._allegro(), infakt, AsyncMock())

        assert out.startswith("✅")
        assert "nie udało się wygenerować linku" in out

    async def test_a_timed_out_issuance_is_recorded_so_it_is_not_nagged_again(self):
        from unittest.mock import AsyncMock

        allegro, infakt = self._allegro(), self._infakt()
        infakt.create_invoice.side_effect = TimeoutError("still pending")
        record = AsyncMock()
        await self._issue(allegro, infakt, record)

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

    async def test_a_known_issuance_is_reported_not_reissued(self):
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
        assert "nie wystawiam drugiej" in out
        assert "FV/9/2026" in out

    async def test_the_earlier_invoice_is_not_attached_behind_the_sellers_back(self):
        """"Wystaw" for the second time is still not permission to show the
        buyer a document nobody has checked."""
        from unittest.mock import AsyncMock, patch

        import services.infakt_service as infakt_service

        allegro, infakt = self._allegro(), self._infakt()
        with patch.object(infakt_service, "build_invoice_payload", return_value={}), \
             patch.object(infakt_service.InfaktService, "get_instance", return_value=infakt), \
             patch("services.invoice_ledger.get_record",
                   AsyncMock(return_value={"invoice_uuid": "inv-OLD", "number": "FV/9/2026"})), \
             patch("services.invoice_ledger.record_issued", AsyncMock()):
            out = await infakt_service.issue_invoice_for_order(allegro, "ORD-1", is_production=False)

        allegro.create_order_invoice_record.assert_not_awaited()
        allegro.upload_order_invoice_file.assert_not_awaited()
        assert "dołącz fakturę do zamówienia `ORD-1`" in out
