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


def test_every_tool_has_a_case():
    assert TOOL_NAMES - COVERED_TOOLS == set(), "tools with no test case"


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


async def test_issue_invoice_checks_the_infakt_task_before_sleeping():
    """A task that is already done must cost exactly one status call and no
    up-front poll_interval wait."""
    result = await run_case(CASES_BY_ID["issue_invoice_for_order"])
    status_calls = [c for c in result["api_calls"] if "async/invoices/status" in c]
    assert len(status_calls) == 1, status_calls
    assert result["duration_ms"] < 1000, f"still waiting up front: {result['duration_ms']} ms"
