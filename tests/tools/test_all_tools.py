from __future__ import annotations

"""End-to-end smoke tests for every Allegro-agent tool.

Unlike tests/unit/test_allegro_tools.py (which only validates the JSON tool
schemas), these actually EXECUTE each tool through AllegroAgent._execute_tool
against fake Allegro/inFakt HTTP transports, and assert on what the seller
would see in the chat.
"""

import pytest

from agents.allegro.allegro_tools import ALLEGRO_TOOLS
from tests.tools.cases import CASES, CASES_BY_ID, COVERED_TOOLS, Case
from tests.tools.runner import run_case

# ask_clarifying_question never reaches _execute_tool: it is an intent
# signal, not a data tool — AllegroAgent.run() short-circuits on it and
# returns the question itself, so there is no tool output to exercise here.
NON_DISPATCH_TOOLS = {"ask_clarifying_question"}
TOOL_NAMES = {t["function"]["name"] for t in ALLEGRO_TOOLS} - NON_DISPATCH_TOOLS

# Phrases the agent emits when a tool blew up — never acceptable output.
ERROR_MARKERS = (
    "An internal error occurred",
    "Unknown tool:",
    "Sesja Allegro wygasła",
)


# ask_clarifying_question never reaches _dispatch: AllegroAgent.run() answers it
# with the model's own question and returns (see the clarify_call short-circuit),
# so there is no tool execution for a case here to drive. Its behavior is covered
# in tests/unit/test_allegro_agent_run.py instead.
NOT_DISPATCHED = {"ask_clarifying_question"}


def test_every_tool_has_a_case():
    assert TOOL_NAMES - COVERED_TOOLS - NOT_DISPATCHED == set(), "tools with no test case"


def test_no_case_targets_an_unknown_tool():
    assert COVERED_TOOLS - TOOL_NAMES == set(), "cases for tools that no longer exist"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
async def test_tool_case(case: Case):
    result = await run_case(case)

    assert result["error"] is None, f"{case.id} raised: {result['error']}"
    assert result["output"].strip(), f"{case.id} returned empty output"
    for marker in ERROR_MARKERS:
        assert marker not in result["output"], f"{case.id} returned an error message: {marker}"
    assert not result["missing_expectations"], (
        f"{case.id} is missing {result['missing_expectations']} in:\n{result['output'][:1500]}"
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
async def test_tool_case_hits_the_api(case: Case):
    """A data tool must actually call Allegro; a UI-only tool must not.

    The exceptions are the monitoring toggles (pure UI state) and inputs the
    agent rejects locally — neither should cost an API round-trip.
    """
    result = await run_case(case)
    ui_only = case.no_api or case.tool.startswith(("suggest_", "disable_"))
    if ui_only:
        assert not result["api_calls"], f"{case.id} unexpectedly called {result['api_calls']}"
    else:
        assert result["api_calls"], f"{case.id} never reached the Allegro/inFakt API"


# ── Regressions found by reading the captured screenshots ────────────────────


async def test_preview_invoices_keeps_the_id_placeholder():
    """Chat replies render as markdown, so a bare <id> is parsed as an HTML tag
    and dropped — the hint has to survive as literal text."""
    result = await run_case(CASES_BY_ID["preview_pending_invoices"])
    assert "`<id>`" in result["output"]
    assert "dla zamówienia `<id>`" in result["output"]


async def test_sales_summary_fee_breakdown_sits_under_the_fee_line():
    """The indented breakdown nests under the preceding bullet, so the fee types
    must follow "Łączne opłaty", not "Zwroty/rabaty"."""
    out = (await run_case(CASES_BY_ID["get_sales_summary"]))["output"]
    fees = out.index("- Łączne opłaty:")
    refunds = out.index("- Zwroty/rabaty:")
    first_fee_type = out.index("  - Prowizja od sprzedaży")
    first_refund_type = out.index("  - Zwrot prowizji")
    assert fees < first_fee_type < refunds < first_refund_type


async def test_order_listings_answer_in_one_shape():
    """get_new_orders / get_orders / get_orders_delivery are one implementation
    behind three intent presets (AllegroAgent._ORDERS_PRESETS), so the seller
    must get the same plain-text bullet block from all of them. They used to
    differ: "nowe zamówienia" came back as chat text while "zamówienia do
    wysłania" came back as a markdown table the PWA hides behind the document
    viewer."""
    fields = ("- Zamawiający:", "- Status:", "- Wysyłka do:", "- Rodzaj dostawy:",
              "- Ilość:", "- Wartość:", "- Link:")
    for case_id in ("get_new_orders", "get_orders", "get_orders_delivery"):
        result = await run_case(CASES_BY_ID[case_id])
        assert result["format"] == "chat", f"{case_id} is not plain text"
        assert "|---" not in result["output"], f"{case_id} rendered a markdown table"
        for field in fields:
            assert field in result["output"], f"{case_id} is missing {field}"


async def test_payment_period_filter_keeps_the_buyer_login():
    """A payment-time filter is applied client-side, so the fetch behind it used
    to be rebuilt from scratch — status forced to READY_FOR_PROCESSING, a fixed
    "last 7 days" window, and every other filter dropped. A question about one
    account ("czy w tym roku kupował ode mnie ktoś z konta np1988") therefore
    came back as the whole store's week with the login silently ignored."""
    from tests.tools.harness import tool_harness

    async with tool_harness() as h:
        out = await h.run("get_orders", {
            "buyer_login": "anna.kowalska88",
            "paid_after_local": "2026-01-01 00:00",
            "count_only": True,
        })
        order_calls = [q for m, p, q in h.allegro_api.calls if p == "/order/checkout-forms"]

    assert "od kupującego **anna.kowalska88**" in out
    assert order_calls and all(q.get("buyer.login") == ["anna.kowalska88"] for q in order_calls), order_calls
    # The window is derived from the requested period, not from "now minus a week".
    assert all(q["lineItems.boughtAt.gte"][0] < "2026-01-01" for q in order_calls), order_calls


async def test_phone_lookup_survives_a_listing_without_phone_numbers():
    """Allegro trims optional fields on the order LIST endpoint. If the phone
    numbers are among them, every "czy mam klienta z tym numerem" would come
    back as a confident "nie" — so the lookup re-fetches the orders one by one
    (real AllegroService, real parsing) before concluding anything."""
    from tests.tools.harness import tool_harness

    async with tool_harness(trim_listed_phones=True) as h:
        out = await h.run("find_buyer_by_contact", {"phone": "+48 880 197 834"})
        detail_calls = [
            p for m, p, q in h.allegro_api.calls
            if p.startswith("/order/checkout-forms/") and not p.endswith("/invoices")
        ]

    assert out.startswith("**Tak —"), out
    assert "Kawa i Spółka" in out
    assert detail_calls, "the listing had no phones and nothing was re-fetched"


async def test_issue_invoice_checks_the_infakt_task_before_sleeping():
    """A task that is already done must cost exactly one status call and no
    up-front poll_interval wait."""
    result = await run_case(CASES_BY_ID["issue_invoice_for_order"])
    status_calls = [c for c in result["api_calls"] if "async/invoices/status" in c]
    assert len(status_calls) == 1, status_calls
    assert result["duration_ms"] < 1000, f"still waiting up front: {result['duration_ms']} ms"


async def test_issue_invoice_waits_out_infakts_mid_processing_status():
    """inFakt answers 140 "Zlecenie jest w trakcie przetwarzania" while it is
    still building the invoice. Only 100 counted as pending, so a task caught
    mid-processing came back to the seller as "❌ Nie udało się wystawić
    faktury" for an invoice inFakt then created anyway. The task has to be
    re-checked until it actually resolves."""
    from unittest.mock import patch

    from tests.tools import dataset as ds
    from tests.tools.harness import tool_harness

    async with tool_harness() as h:
        h.infakt_api.pending_codes = [100, 140, 140]
        with patch("asyncio.sleep"):  # don't spend 3 × poll_interval on this
            out = await h.run("issue_invoice_for_order", {"order_id": ds.ORD_1})

    assert out.startswith("✅"), out
    assert not h.infakt_api.pending_codes, "the task was not polled to completion"
    assert "140" not in out
