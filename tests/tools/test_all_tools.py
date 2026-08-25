from __future__ import annotations

"""End-to-end smoke tests for every Allegro-agent tool.

Unlike tests/unit/test_allegro_tools.py (which only validates the JSON tool
schemas), these actually EXECUTE each tool through AllegroAgent._execute_tool
against fake Allegro/inFakt HTTP transports, and assert on what the seller
would see in the chat.
"""

import pytest

from agents.allegro.allegro_tools import ALLEGRO_TOOLS
from tests.tools.cases import CASES, COVERED_TOOLS, Case
from tests.tools.runner import run_case

TOOL_NAMES = {t["function"]["name"] for t in ALLEGRO_TOOLS}

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
