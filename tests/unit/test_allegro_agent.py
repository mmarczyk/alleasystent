"""Unit tests for the get_new_orders interpret-bypass optimization in
agents/allegro/allegro_agent.py — see AllegroAgent.run() for the rationale:
get_new_orders' dispatch output already matches _TOOL_SPECIFIC_INSTRUCTIONS
verbatim for a Polish query, so the interpret LLM call is pure passthrough
and can be skipped entirely.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.allegro.allegro_agent import _is_confidently_polish


class TestIsConfidentlyPolish:
    def test_polish_diacritics_detected(self):
        assert _is_confidently_polish("jakie mam nowe zamówienia")
        assert _is_confidently_polish("pokaż moje oferty")

    def test_ascii_only_english_not_detected(self):
        assert not _is_confidently_polish("show me my new orders")

    def test_ascii_only_polish_not_detected(self):
        """Conservative by design: missing this optimization is fine, wrongly
        assuming Polish and skipping translation for a real English query is not."""
        assert not _is_confidently_polish("ile mam zamowien")


def _tool_call(name: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = "c1"
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    tc.model_dump.return_value = {
        "id": "c1",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return tc


def _tool_select_response(tool_calls: list) -> MagicMock:
    msg = MagicMock()
    msg.tool_calls = tool_calls
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _no_more_tools_response() -> MagicMock:
    """The chained tool loop always offers a second, tool_choice='auto' round
    after the first — this is what the model returns to end it there."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def _make_agent():
    from agents.allegro.allegro_agent import AllegroAgent

    with patch("agents.base_agent.AsyncOpenAI"), \
         patch("agents.allegro.allegro_agent.AllegroService") as MockService:
        mock_service = MagicMock()
        mock_service._tokens = MagicMock()
        mock_service._tokens.is_expired.return_value = False
        MockService.get_instance.return_value = mock_service
        agent = AllegroAgent()
    return agent


class TestGetNewOrdersInterpretBypass:
    @pytest.mark.asyncio
    async def test_polish_query_skips_interpret_call(self):
        # "jakie mam nowe zamówienia" is also a deterministic-dispatch match
        # (see agents/allegro/deterministic_dispatch.py) — the tool-select
        # LLM call is skipped too, on top of the interpret bypass this class
        # is about, so call_count is 0 rather than needing a mocked response.
        agent = _make_agent()
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )

        raw_text = "**Zamówienie** `X`\n- Zamawiający: **jan_kowalski**"
        agent._dispatch = AsyncMock(return_value=raw_text)

        response = await agent.run("jakie mam nowe zamówienia")

        assert response.text == raw_text
        assert agent._client.chat.completions.create.call_count == 0
        assert "interpret_llm" not in response.metadata["perf_stages"]
        assert response.metadata["tools"] == ["get_new_orders"]

    @pytest.mark.asyncio
    async def test_english_query_still_uses_interpret_call(self):
        agent = _make_agent()
        tool_select_resp = _tool_select_response([_tool_call("get_new_orders", {})])

        interp_msg = MagicMock()
        interp_msg.tool_calls = None
        interp_msg.content = "**Order** `X`\n- Buyer: **jan_kowalski**"
        interp_choice = MagicMock()
        interp_choice.message = interp_msg
        interp_resp = MagicMock()
        interp_resp.choices = [interp_choice]

        agent._client.chat.completions.create = AsyncMock(
            side_effect=[tool_select_resp, _no_more_tools_response(), interp_resp]
        )
        agent._dispatch = AsyncMock(
            return_value="**Zamówienie** `X`\n- Zamawiający: **jan_kowalski**"
        )

        response = await agent.run("show me my new orders")

        assert response.text == "**Order** `X`\n- Buyer: **jan_kowalski**"
        assert agent._client.chat.completions.create.call_count == 3
        assert "interpret_llm" in response.metadata["perf_stages"]

    @pytest.mark.asyncio
    async def test_count_only_bypasses_too(self):
        """count_only used to be the one get_new_orders case that still needed
        the interpret call, because the dispatch said "Liczba nowych zamówień: 3."
        and the tool instruction asked for "Masz 3 nowe zamówienia." — the
        dispatch now words it that way itself, so there is nothing left to do."""
        agent = _make_agent()
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )
        agent._dispatch = AsyncMock(return_value="Masz **3** nowe zamówienia.")

        response = await agent.run("ile mam nowych zamówień")

        assert response.text == "Masz **3** nowe zamówienia."
        assert agent._client.chat.completions.create.call_count == 0

    @pytest.mark.asyncio
    async def test_multi_tool_turn_still_uses_interpret_call(self):
        """A turn calling get_new_orders alongside another tool still needs the
        LLM to weave the results together — passthrough only applies when it
        was the ONLY tool called."""
        agent = _make_agent()
        tool_select_resp = _tool_select_response([
            _tool_call("get_new_orders", {}),
            _tool_call("get_account_info", {}),
        ])

        interp_msg = MagicMock()
        interp_msg.tool_calls = None
        interp_msg.content = "combined answer"
        interp_choice = MagicMock()
        interp_choice.message = interp_msg
        interp_resp = MagicMock()
        interp_resp.choices = [interp_choice]

        agent._client.chat.completions.create = AsyncMock(
            side_effect=[tool_select_resp, _no_more_tools_response(), interp_resp]
        )
        agent._dispatch = AsyncMock(side_effect=["orders text", "account text"])

        response = await agent.run("nowe zamówienia i moje konto")

        assert response.text == "combined answer"
        assert agent._client.chat.completions.create.call_count == 3


class TestPassthroughGeneralizedToOtherTools:
    """The get_new_orders bypass generalizes to every tool in
    _PASSTHROUGH_TOOLS — which, now that each tool renders its own finished
    view in _dispatch, is every tool there is. Spot-check a list tool and an
    action-toggle tool, then the one path that still reaches the interpret
    call: a non-Polish query."""

    @pytest.mark.asyncio
    async def test_get_new_returns_count_only_also_bypasses(self):
        # Also a deterministic-dispatch match — see comment on
        # test_polish_query_skips_interpret_call in test_allegro_agent.py.
        agent = _make_agent()
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )
        agent._dispatch = AsyncMock(return_value="Masz **2** zwroty.")

        response = await agent.run("czy mam jakieś zwroty")

        assert response.text == "Masz **2** zwroty."
        assert agent._client.chat.completions.create.call_count == 0

    @pytest.mark.asyncio
    async def test_monitoring_toggle_bypasses(self):
        # Also a deterministic-dispatch match — see comment above.
        agent = _make_agent()
        agent._client.chat.completions.create = AsyncMock(
            side_effect=AssertionError("no LLM call expected — deterministic match")
        )
        status_block = (
            "💡 Mogę automatycznie sprawdzać nowe zamówienia...\n\n"
            '<button onclick="OrderMonitor.enable()">🔔 Włącz</button>'
        )
        agent._dispatch = AsyncMock(return_value=status_block)

        response = await agent.run("włącz monitoring zamówień")

        assert response.text == status_block
        assert agent._client.chat.completions.create.call_count == 0

    @pytest.mark.asyncio
    async def test_english_query_still_uses_interpret_call(self):
        """Every tool renders its own view now, so nothing is left OUT of
        _PASSTHROUGH_TOOLS — the interpret call survives for the one job the
        rendering can't do: answering a non-Polish question, where it
        translates the finished view's labels rather than rebuilding it."""
        agent = _make_agent()
        tool_select_resp = _tool_select_response(
            [_tool_call("get_message_threads", {})]
        )

        interp_msg = MagicMock()
        interp_msg.tool_calls = None
        interp_msg.content = "- **jan_kowalski** — unread"
        interp_choice = MagicMock()
        interp_choice.message = interp_msg
        interp_resp = MagicMock()
        interp_resp.choices = [interp_choice]

        agent._client.chat.completions.create = AsyncMock(
            side_effect=[tool_select_resp, _no_more_tools_response(), interp_resp]
        )
        agent._dispatch = AsyncMock(
            return_value="- **jan_kowalski** — 🔴 nieprzeczytana — wątek `t1`"
        )

        response = await agent.run("show me my messages")

        assert response.text == "- **jan_kowalski** — unread"
        assert agent._client.chat.completions.create.call_count == 3


class TestGetActiveOffersDispatch:
    """_dispatch's get_active_offers branch builds the finished reply in Python
    (heading + markdown table + trailing summary) instead of asking the interpret
    LLM to turn a bullet list into a table — see AllegroAgent._render_offers_table.
    The layout is what splits the answer in the PWA: summary sentence in the chat
    bubble, table in the document viewer (web/js/app.js _tablePreview/_docTitle)."""

    def _offer(self, oid, name, price, stock, ended=False):
        offer = {
            "id": oid,
            "name": name,
            "sellingMode": {"price": {"amount": str(price), "currency": "PLN"}},
            "stock": {"available": stock},
        }
        if ended:
            offer["_ended"] = True
        return offer

    def _make_agent_with_offers(self, active, ended=None):
        agent = _make_agent()
        agent._allegro.get_all_offers = AsyncMock(
            side_effect=lambda status="ACTIVE": active if status == "ACTIVE" else (ended or [])
        )
        agent._allegro.get_offers = AsyncMock(
            side_effect=lambda publication_status="ACTIVE", **kw: (
                (active, len(active)) if publication_status == "ACTIVE" else (ended or [], len(ended or []))
            )
        )
        return agent

    @pytest.mark.asyncio
    async def test_heading_table_and_trailing_summary(self):
        agent = self._make_agent_with_offers(
            active=[
                self._offer("111", "Czajnik Bosch", 109.0, 27),
                self._offer("222", "Toster Russell Hobbs", 179.0, 0),
                self._offer("333", "Czajnik Bosch", 109.0, 3),
            ],
            ended=[self._offer("444", "Żelazko Philips", 329.0, 0)],
        )

        result = await agent._dispatch("get_active_offers", {})
        lines = result.splitlines()

        # Heading first — it names the document tab; without it the tab title is
        # a garbled '| Nazwa | Stan…' row.
        assert lines[0] == "# Aktywne oferty"
        assert "| Nazwa | Stan (szt.) | Cena | ID ofert |" in lines
        assert "| --- | ---: | ---: | --- |" in lines
        # Rows ascending by stock, same-name offers aggregated, ended-and-sold-out
        # offers flagged rather than dropped.
        rows = [ln for ln in lines if ln.startswith("| ") and "---" not in ln][1:]
        assert rows == [
            "| Toster Russell Hobbs | 0 | 179,00 PLN | `222` |",
            "| Żelazko Philips *(zakończona — wyprzedana)* | 0 | 329,00 PLN | `444` |",
            "| Czajnik Bosch | 30 | 109,00 PLN | `111`, `333` |",
        ]
        # The summary is LAST — that is the half the chat bubble shows.
        assert lines[-1].startswith("Masz obecnie **3** aktywne oferty")
        assert "w tabeli **3** produkty" in lines[-1]
        assert "W tym **1** zakończona przez wyprzedanie" in lines[-1]

    @pytest.mark.asyncio
    async def test_summary_omits_ended_clause_when_there_are_none(self):
        agent = self._make_agent_with_offers(active=[self._offer("111", "Czajnik Bosch", 109.0, 27)])

        result = await agent._dispatch("get_active_offers", {})

        assert result.splitlines()[-1] == (
            "Masz obecnie **1** aktywną ofertę — w tabeli **1** produkt "
            "po zsumowaniu ofert o tej samej nazwie."
        )

    @pytest.mark.asyncio
    async def test_name_filter_is_named_in_the_summary(self):
        agent = self._make_agent_with_offers(
            active=[self._offer("111", "Ekspres DeLonghi", 1249.0, 4)]
        )

        result = await agent._dispatch("get_active_offers", {"name": "ekspres"})

        assert "(filtr nazwy: „ekspres”)" in result.splitlines()[-1]

    @pytest.mark.asyncio
    async def test_pipe_in_offer_name_is_escaped(self):
        agent = self._make_agent_with_offers(
            active=[self._offer("111", "Czajnik | Bosch", 109.0, 5)]
        )

        result = await agent._dispatch("get_active_offers", {})

        assert "| Czajnik \\| Bosch | 5 | 109,00 PLN | `111` |" in result

    @pytest.mark.asyncio
    async def test_no_offers_answers_in_one_sentence_without_a_table(self):
        agent = self._make_agent_with_offers(active=[])

        assert await agent._dispatch("get_active_offers", {}) == "Brak aktywnych ofert."
        assert await agent._dispatch("get_active_offers", {"name": "ekspres"}) == (
            "Brak aktywnych ofert pasujących do „ekspres”."
        )

    @pytest.mark.asyncio
    async def test_polish_query_skips_the_interpret_call(self):
        """get_active_offers is in _PASSTHROUGH_TOOLS now that its dispatch
        output is the finished answer — a Polish query must reach the user
        verbatim, table and summary intact, with no interpret round to retype
        (and truncate) hundreds of rows."""
        agent = _make_agent()
        table = (
            "# Aktywne oferty\n\n"
            "| Nazwa | Stan (szt.) | Cena | ID ofert |\n"
            "| --- | ---: | ---: | --- |\n"
            "| Czajnik Bosch | 27 | 109,00 PLN | `111` |\n\n"
            "Masz obecnie **1** aktywną ofertę — w tabeli **1** produkt "
            "po zsumowaniu ofert o tej samej nazwie."
        )
        agent._client.chat.completions.create = AsyncMock(
            side_effect=[_tool_select_response([_tool_call("get_active_offers", {})]),
                         _no_more_tools_response()]
        )
        agent._dispatch = AsyncMock(return_value=table)

        response = await agent.run("pokaż moje aktywne oferty")

        assert response.text == table
        assert response.metadata["output_format"] == "table"
        assert "interpret_llm" not in response.metadata["perf_stages"]
        # Tool-select round + the "anything else?" round, and no interpret call.
        assert agent._client.chat.completions.create.call_count == 2


class TestEveryToolRendersItsOwnView:
    """The design rule this file's bypass tests rest on: the model picks WHICH
    tool runs and (for a non-Polish query) translates labels — it never builds
    the output view out of raw data. Every tool's dispatch returns the finished
    answer, so every tool is a passthrough tool."""

    def test_no_tool_is_left_for_the_model_to_format(self):
        from agents.allegro.allegro_agent import _PASSTHROUGH_TOOLS, _RENDERED_VIEW_TOOLS
        from agents.allegro.allegro_tools import ALLEGRO_TOOLS, TOOL_OUTPUT_FORMAT

        declared = {t["function"]["name"] for t in ALLEGRO_TOOLS}
        assert declared == set(TOOL_OUTPUT_FORMAT)
        assert declared <= _RENDERED_VIEW_TOOLS
        assert declared <= _PASSTHROUGH_TOOLS

    def test_the_interpret_instruction_never_asks_for_a_rebuild(self):
        from agents.allegro.allegro_agent import _RENDERED_VIEW_INSTRUCTION

        assert "PRZEPISZ BEZ ZMIAN" in _RENDERED_VIEW_INSTRUCTION
        # No "build a table out of the data above" wording survives anywhere.
        assert "Zbuduj tabelę" not in _RENDERED_VIEW_INSTRUCTION


class TestRenderedViews:
    """Spot-checks on the Python renderers behind the tool outputs — the shapes
    the PWA parses (web/js/app.js): a '# ' heading names the document tab, a
    table's trailing sentence becomes the chat bubble, a document/dashboard's
    first line under the heading does."""

    def _agent(self):
        return _make_agent()

    def test_md_table_escapes_pipes_and_aligns_columns(self):
        agent = self._agent()

        lines = agent._md_table(["A", "B"], [["x|y", 3], ["", None]], align="lr")

        assert lines[0] == "| A | B |"
        assert lines[1] == "| --- | ---: |"
        assert lines[2] == "| x\\|y | 3 |"
        # Empty cells become an em dash rather than collapsing the row.
        assert lines[3] == "| — | None |"

    @pytest.mark.asyncio
    async def test_order_listing_is_a_bullet_list_never_a_table(self):
        """Orders are the one listing that deliberately stays plain chat text —
        a markdown table would be hidden behind the document view in the PWA
        (see _ORDERS_PRESETS / _order_bullet). Rendered in Python all the same."""
        from models.allegro import AllegroOrder, AllegroOrderLine

        order = AllegroOrder(
            order_id="abc-123",
            buyer_login="jan_kowalski",
            buyer_email="jan@example.com",
            status="BOUGHT",
            fulfillment_status="NEW",
            total_price=189.98,
            currency="PLN",
            created_at="2026-08-27T10:15:00Z",
            paid_at="2026-08-27T10:20:00Z",
            delivery={"method": {"name": "InPost Paczkomaty"}},
            line_items=[AllegroOrderLine(offer_id="1", offer_name="Sweter", quantity=2, price=94.99)],
        )
        agent = self._agent()
        agent._allegro.get_orders = AsyncMock(return_value=[order])

        result = await agent._dispatch("get_orders", {})

        assert result.startswith("**Zamówienie** `abc-123`")
        assert "|" not in result
        assert "- Status: **Nowe**" in result
        assert "- Wysyłka do:" in result

    @pytest.mark.asyncio
    async def test_order_listing_count_only_is_the_final_sentence(self):
        agent = self._agent()
        agent._allegro.get_orders = AsyncMock(return_value=[object(), object()])

        assert (await agent._dispatch("get_orders", {"count_only": True})).startswith(
            "Masz łącznie **2** zamówienia."
        )
        assert (await agent._dispatch("get_orders_delivery", {"count_only": True})).startswith(
            "Do wysłania: **2** zamówienia."
        )

    def test_dashboard_chart_json_matches_the_bullets(self):
        agent = self._agent()

        out = agent._render_dashboard(
            title="Podsumowanie ofert",
            lead="Aktywnych ofert: **3**.",
            sections=[("Stany magazynowe", {"0 szt.": 2, "1–9 szt.": 0, "10+ szt.": 1}, "Wykres")],
        )

        assert out.startswith("## Podsumowanie ofert")
        # Empty buckets are dropped from both the bullets and the chart.
        assert "- 0 szt.: **2** oferty" in out
        assert "1–9 szt." not in out
        spec = json.loads(out.split("```chart\n")[1].split("\n```")[0])
        assert spec["labels"] == ["0 szt.", "10+ szt."]
        assert spec["series"][0]["data"] == [2, 1]

    @pytest.mark.asyncio
    async def test_new_orders_count_only_is_the_final_sentence(self):
        agent = self._agent()
        agent._allegro.get_orders = AsyncMock(return_value=[object(), object(), object()])
        agent._monitoring_status_block = AsyncMock(return_value="")

        result = await agent._dispatch("get_new_orders", {"count_only": True})

        assert result.startswith("Masz **3** nowe zamówienia.")

    @pytest.mark.asyncio
    async def test_every_count_only_answer_is_one_finished_sentence(self):
        """One helper words them all (see _count_sentence) — no "Liczba X: N."
        label survives anywhere, because nothing downstream rewords it."""
        agent = self._agent()
        agent._returns_monitoring_status_block = AsyncMock(return_value="")
        agent._allegro.get_customer_returns = AsyncMock(return_value=[object()] * 2)
        agent._allegro.get_issues = AsyncMock(return_value=[object()])
        agent._allegro.get_message_threads = AsyncMock(return_value=[
            {"id": "t1", "read": False, "interlocutor": {"login": "jan"}},
        ])

        assert (await agent._dispatch("get_new_returns", {"count_only": True})).startswith(
            "Masz **2** zwroty."
        )
        assert (await agent._dispatch("get_returns_to_process", {"count_only": True})).startswith(
            "Do obsłużenia masz **2** zwroty."
        )
        assert (await agent._dispatch("get_new_complaints", {"count_only": True})).startswith(
            "Masz **1** reklamację."
        )
        assert (await agent._dispatch("get_message_threads", {"count_only": True})) == (
            "Masz **1** nową wiadomość (od: jan). Pokazać szczegóły?"
        )

    def test_count_sentence_uses_polish_plural_forms_and_a_zero_case(self):
        agent = self._agent()
        forms = ("zwrot", "zwroty", "zwrotów")

        assert agent._count_sentence(0, "Masz", forms, none="Brak zwrotów.") == "Brak zwrotów."
        assert agent._count_sentence(1, "Masz", forms, none="x") == "Masz **1** zwrot."
        assert agent._count_sentence(3, "Masz", forms, none="x") == "Masz **3** zwroty."
        assert agent._count_sentence(12, "Masz", forms, none="x") == "Masz **12** zwrotów."
        assert agent._count_sentence(2, "Masz", forms, none="x", scope=" (sierpień)") == (
            "Masz **2** zwroty (sierpień)."
        )

    @pytest.mark.asyncio
    async def test_no_new_orders_count_only(self):
        agent = self._agent()
        agent._allegro.get_orders = AsyncMock(return_value=[])
        agent._monitoring_status_block = AsyncMock(return_value="")

        result = await agent._dispatch("get_new_orders", {"count_only": True})

        assert result.startswith("Nie masz nowych zamówień.")


class TestGetOrderDetailsDispatch:
    """_dispatch's get_order_details branch now builds the final, ready-to-
    display plain-text bullet list directly in Python instead of handing
    raw fields to the interpret LLM — _TOOL_SPECIFIC_INSTRUCTIONS["get_order_details"]
    already fully prescribed the shape, so there was no real judgment left
    for the LLM to apply. See _PASSTHROUGH_TOOLS."""

    def _make_order(self, **overrides):
        from models.allegro import AllegroOrder, AllegroOrderLine
        defaults = dict(
            order_id="abc-123",
            buyer_login="jan_kowalski",
            buyer_email="jan@example.com",
            status="BOUGHT",
            fulfillment_status="READY_FOR_PROCESSING",
            total_price=189.98,
            currency="PLN",
            created_at="2026-08-27T10:15:00Z",
            paid_at="2026-08-27T10:20:00Z",
            delivery={"method": {"name": "InPost Paczkomaty"}, "trackingCode": "PL123456789"},
            line_items=[
                AllegroOrderLine(offer_id="111", offer_name="Sweter wełniany M", quantity=1, price=129.99),
                AllegroOrderLine(offer_id="222", offer_name="Skarpety wełniane 3-pak", quantity=1, price=59.99),
            ],
            invoice_required=False,
        )
        defaults.update(overrides)
        return AllegroOrder(**defaults)

    def _make_agent_with_order(self, order, billing_entries=None, existing_invoices=None):
        agent = _make_agent()
        agent._allegro.get_order = AsyncMock(return_value=order)
        agent._allegro.get_billing_entries_for_order = AsyncMock(return_value=billing_entries or [])
        agent._allegro.get_order_invoices = AsyncMock(return_value=existing_invoices or [])
        return agent

    @pytest.mark.asyncio
    async def test_plain_text_bullet_list_no_headers_or_code_fences(self):
        order = self._make_order()
        agent = self._make_agent_with_order(order)

        result = await agent._dispatch("get_order_details", {"order_id": "abc-123"})

        assert "```" not in result
        assert "#" not in result
        assert "- Zamówienie: `abc-123`" in result
        assert "- Kupujący: jan_kowalski" in result
        assert "- Wartość: 189,98 PLN" in result
        assert "- Produkty:" in result
        assert "  - Sweter wełniany M (ID: 111): 1 × 129,99 PLN" in result
        assert "- Dostawa:" in result
        assert "  - Metoda: InPost Paczkomaty" in result
        assert "  - Tracking: PL123456789" in result

    @pytest.mark.asyncio
    async def test_no_billing_entries_omits_billing_section(self):
        order = self._make_order()
        agent = self._make_agent_with_order(order, billing_entries=[])

        result = await agent._dispatch("get_order_details", {"order_id": "abc-123"})

        assert "Rozliczenie" not in result

    @pytest.mark.asyncio
    async def test_every_billing_entry_shown_separately(self):
        order = self._make_order()
        billing_entries = [
            {"value": {"amount": "-12.99"}, "type": {"description": "Prowizja od sprzedaży"}, "offer": {"name": "Sweter"}, "occurredAt": "2026-08-27T00:00:00Z"},
            {"value": {"amount": "-6.00"}, "type": {"description": "Prowizja od sprzedaży"}, "offer": {"name": "Skarpety"}, "occurredAt": "2026-08-27T00:00:00Z"},
            {"value": {"amount": "-1.00"}, "type": {"description": "Opłata za wystawienie oferty"}, "occurredAt": "2026-08-27T00:00:00Z"},
        ]
        agent = self._make_agent_with_order(order, billing_entries=billing_entries)

        result = await agent._dispatch("get_order_details", {"order_id": "abc-123"})

        assert "- Rozliczenie:" in result
        assert result.count("Prowizja od sprzedaży") == 2
        assert "Opłata za wystawienie oferty" in result
        assert "Suma opłat: -19.99 PLN" in result
        assert "Zysk netto: 169.99 PLN" in result

    @pytest.mark.asyncio
    async def test_invoice_status_variants(self):
        cases = [
            (dict(invoice_required=False), [], "Kupujący nie poprosił o fakturę."),
            (dict(invoice_required=True), [{"id": "inv-1"}], "faktura już wystawiona."),
            (dict(invoice_required=True), [], "NIE WYSTAWIONO jeszcze faktury."),
        ]
        for order_overrides, existing_invoices, expected_substr in cases:
            order = self._make_order(**order_overrides)
            agent = self._make_agent_with_order(order, existing_invoices=existing_invoices)

            result = await agent._dispatch("get_order_details", {"order_id": "abc-123"})

            assert expected_substr in result, result


class TestOrdersDueToday:
    """get_orders_due_today is the deadline view: what still has to be handed
    to the carrier by the end of today. Unlike the other order presets it is
    defined by exclusion (see _ORDERS_PRESETS), so these pin what it drops."""

    @staticmethod
    def _order(order_id: str, fulfillment_status: str, dispatch_to: str):
        from models.allegro import AllegroOrder, AllegroOrderLine

        return AllegroOrder(
            order_id=order_id,
            buyer_login="kupujacy",
            buyer_email="k@example.com",
            status="READY_FOR_PROCESSING",
            fulfillment_status=fulfillment_status,
            total_price=99.0,
            currency="PLN",
            created_at="2026-08-27T10:00:00Z",
            paid_at="2026-08-27T10:05:00Z",
            delivery={"method": {"id": "dpd", "name": "Kurier DPD"}},
            line_items=[AllegroOrderLine(offer_id="1", offer_name="Produkt", quantity=1, price=99.0)],
            invoice_required=False,
            dispatch_to=dispatch_to,
        )

    @staticmethod
    def _deadline(days: int, hhmm: str) -> str:
        """A dispatch deadline at a Warsaw-local wall-clock time, built through
        the same conversion the filter uses. Deliberately anchored to calendar
        days rather than "now + N hours": the tool's cut-off is the end of the
        local day, so an offset-based fixture would land on the wrong side of
        it when the suite happens to run near midnight."""
        from datetime import date, timedelta

        from agents.allegro.allegro_agent import AllegroAgent

        day = (date.today() + timedelta(days=days)).isoformat()
        return AllegroAgent._local_to_utc(f"{day} {hhmm}")

    def _agent_with(self, orders):
        agent = _make_agent()
        agent._allegro.get_orders = AsyncMock(return_value=orders)
        agent._allegro.get_carriers = AsyncMock(return_value=[{"id": "dpd", "name": "Kurier DPD"}])
        return agent

    def _all_orders(self):
        return [
            self._order("overdue", "NEW", self._deadline(-1, "08:00")),
            self._order("morning", "PROCESSING", self._deadline(0, "09:00")),
            self._order("evening", "READY_FOR_SHIPMENT", self._deadline(0, "23:00")),
            self._order("sent", "SENT", self._deadline(-1, "10:00")),
            self._order("picked-up", "PICKED_UP", self._deadline(-1, "11:00")),
            self._order("cancelled", "CANCELLED", self._deadline(-1, "12:00")),
            self._order("tomorrow", "NEW", self._deadline(1, "10:00")),
            self._order("next-week", "NEW", self._deadline(7, "10:00")),
            self._order("no-deadline", "NEW", ""),
        ]

    @pytest.mark.asyncio
    async def test_lists_everything_still_to_be_handed_over_and_due_today(self):
        agent = self._agent_with(self._all_orders())

        result = await agent._dispatch("get_orders_due_today", {})

        # Not packed yet (NEW) and half-done (PROCESSING) count too — the
        # deadline doesn't care how far along the order is.
        assert "`overdue`" in result and "`morning`" in result and "`evening`" in result

    @pytest.mark.asyncio
    async def test_drops_what_no_longer_has_to_be_handed_over(self):
        agent = self._agent_with(self._all_orders())

        result = await agent._dispatch("get_orders_due_today", {})

        for order_id in ("sent", "picked-up", "cancelled"):
            assert f"`{order_id}`" not in result, f"{order_id} is not waiting to be dispatched"

    @pytest.mark.asyncio
    async def test_drops_deadlines_beyond_today_and_unknown_ones(self):
        agent = self._agent_with(self._all_orders())

        result = await agent._dispatch("get_orders_due_today", {})

        assert "`tomorrow`" not in result and "`next-week`" not in result
        # An unknown deadline must never be rendered as an urgent one
        # (see _dispatch_within).
        assert "`no-deadline`" not in result

    @pytest.mark.asyncio
    async def test_overdue_first_then_by_deadline(self):
        agent = self._agent_with(self._all_orders())

        result = await agent._dispatch("get_orders_due_today", {})

        assert result.index("`overdue`") < result.index("`morning`") < result.index("`evening`")

    @pytest.mark.asyncio
    async def test_count_only_counts_the_same_set(self):
        agent = self._agent_with(self._all_orders())

        result = await agent._dispatch("get_orders_due_today", {"count_only": True})

        assert "Na dziś do wysłania: **3** zamówienia." in result

    @pytest.mark.asyncio
    async def test_empty_result_says_nothing_is_due(self):
        agent = self._agent_with([self._order("next-week", "NEW", self._deadline(7, "10:00"))])

        result = await agent._dispatch("get_orders_due_today", {})

        assert result == "Nic nie czeka na wysyłkę z dzisiejszym terminem."

    @pytest.mark.asyncio
    async def test_explicit_cut_off_overrides_the_end_of_today_default(self):
        from datetime import date, timedelta

        agent = self._agent_with(self._all_orders())
        tomorrow_2359 = (date.today() + timedelta(days=1)).isoformat() + " 23:59"

        result = await agent._dispatch("get_orders_due_today", {"dispatch_before_local": tomorrow_2359})

        assert "`tomorrow`" in result and "`next-week`" not in result

    @pytest.mark.asyncio
    async def test_fetches_a_full_page_because_both_filters_are_client_side(self):
        """The status exclusion and the deadline filter both run on our side,
        so a limit=5 request must still fetch a whole page — otherwise the
        page could be all already-sent orders and the answer empty."""
        agent = self._agent_with(self._all_orders())

        await agent._dispatch("get_orders_due_today", {"limit": 5})

        assert agent._allegro.get_orders.call_args.kwargs["limit"] == 100


class TestBuyersReport:
    """get_buyers is the buyer view of a period: one row per CUSTOMER, grouped
    by NIP → company name → Allegro login, with the company/private-person
    distinction read off the order's VAT-invoice address (the only place
    Allegro states it) — see AllegroAgent._buyers_report."""

    @staticmethod
    def _order(order_id, login, price, *, company="", nip="", first="", last="",
               invoice_required=False, paid_at="2026-03-04T10:00:00Z",
               recipient="", recipient_company=""):
        from models.allegro import AllegroInvoiceBuyer, AllegroOrder, AllegroOrderLine

        recipient_first, _, recipient_last = recipient.partition(" ")
        return AllegroOrder(
            order_id=order_id,
            buyer_login=login,
            buyer_email=f"{login}@example.com",
            status="READY_FOR_PROCESSING",
            fulfillment_status="SENT",
            total_price=price,
            currency="PLN",
            created_at=paid_at,
            paid_at=paid_at,
            delivery={
                "method": {"id": "dpd", "name": "Kurier DPD"},
                "address": {
                    "firstName": recipient_first,
                    "lastName": recipient_last,
                    "companyName": recipient_company,
                },
            } if (recipient or recipient_company) else {},
            line_items=[AllegroOrderLine(offer_id="1", offer_name="Produkt", quantity=1, price=price)],
            invoice_required=invoice_required,
            invoice_buyer=AllegroInvoiceBuyer(
                required=invoice_required,
                company_name=company,
                vat_id=nip,
                first_name=first,
                last_name=last,
            ),
        )

    def _agent_with(self, orders, issued=()):
        agent = _make_agent()
        agent._allegro.get_all_paid_orders_in_period = AsyncMock(return_value=orders)
        agent._allegro.invoices_issued_map = AsyncMock(
            side_effect=lambda ids: {oid: oid in issued for oid in ids}
        )
        return agent

    def _mixed_orders(self):
        return [
            self._order("c1", "anna", 400.0, company="KAWA I SPOLKA", nip="779-244-55-88",
                        invoice_required=True, paid_at="2026-02-01T10:00:00Z"),
            # Same company, a different Allegro account and a re-typed NIP —
            # grouped by the digits, so this is one buyer with two orders, shown
            # under the name and NIP from the more recent of them.
            self._order("c2", "anna.firma", 600.0, company="Kawa i Spółka", nip="7792445588",
                        invoice_required=True, paid_at="2026-05-01T10:00:00Z"),
            self._order("p1", "marek", 899.0, first="Marek", last="Zieliński",
                        invoice_required=True, paid_at="2026-03-01T10:00:00Z",
                        recipient="Marek Zieliński"),
            # No invoice at all — the name can only come from the parcel's
            # delivery address.
            self._order("p2", "kasia", 137.7, paid_at="2026-04-01T10:00:00Z",
                        recipient="Katarzyna Wójcik"),
        ]

    @pytest.mark.asyncio
    async def test_groups_orders_by_buyer_highest_spend_first(self):
        agent = self._agent_with(self._mixed_orders())

        result = await agent._dispatch("get_buyers", {})
        rows = [ln for ln in result.splitlines() if ln.startswith("| ") and "---" not in ln][1:]

        assert rows == [
            "| Kawa i Spółka | Firma | 7792445588 | `anna.firma`, `anna` | 2 | 1000,00 PLN | 01.05.2026 |",
            "| Marek Zieliński | Osoba prywatna | — | `marek` | 1 | 899,00 PLN | 01.03.2026 |",
            "| Katarzyna Wójcik | Osoba prywatna | — | `kasia` | 1 | 137,70 PLN | 01.04.2026 |",
        ]
        assert result.splitlines()[0] == "# Kupujący"
        # The summary is LAST — that is the half the chat bubble shows.
        assert result.splitlines()[-1].startswith("**3** kupujących w okresie")

    @pytest.mark.asyncio
    async def test_buyer_column_names_the_person_not_their_allegro_login(self):
        """The name comes from the invoice address, else from whoever the parcel
        was addressed to — a login like `kasia.w` is a machine handle, and it
        has its own column."""
        agent = self._agent_with([
            self._order("i1", "kasia.w", 100.0, first="Katarzyna", last="Wójcik",
                        invoice_required=True, recipient="Kasia W"),
            self._order("d1", "tomek.nowak", 90.0, recipient="Tomasz Nowak"),
            self._order("f1", "biuro77", 80.0, recipient_company="Biuro Serwis sp. z o.o."),
        ])

        result = await agent._dispatch("get_buyers", {})

        assert "| Katarzyna Wójcik |" in result   # invoice name wins over delivery
        assert "| Tomasz Nowak |" in result       # no invoice → delivery address
        assert "| Biuro Serwis sp. z o.o. |" in result  # delivery company name
        for login in ("| kasia.w |", "| tomek.nowak |", "| biuro77 |"):
            assert login not in result, f"{login} rendered as a buyer name"

    @pytest.mark.asyncio
    async def test_login_is_the_last_resort_when_no_order_names_the_buyer(self):
        agent = self._agent_with([self._order("n1", "anonim22", 50.0)])

        result = await agent._dispatch("get_buyers", {})

        assert "| anonim22 | Osoba prywatna | — | `anonim22` | 1 |" in result

    @pytest.mark.asyncio
    async def test_name_falls_back_to_an_older_order_that_has_one(self):
        """Newest-first, but a nameless newest order must not blank out a buyer
        the seller can identify from their previous one."""
        agent = self._agent_with([
            self._order("new", "kasia", 10.0, paid_at="2026-06-01T10:00:00Z"),
            self._order("old", "kasia", 10.0, paid_at="2026-01-01T10:00:00Z",
                        recipient="Katarzyna Wójcik"),
        ])

        result = await agent._dispatch("get_buyers", {})

        assert "| Katarzyna Wójcik |" in result

    @pytest.mark.asyncio
    async def test_company_filter_keeps_only_buyers_with_invoice_company_data(self):
        agent = self._agent_with(self._mixed_orders())

        result = await agent._dispatch("get_buyers", {"buyer_type": "company"})

        assert "Kawa i Spółka" in result
        assert "Marek Zieliński" not in result and "`kasia`" not in result
        assert result.splitlines()[-1].startswith("**1** kupujący (firmy) w okresie")

    @pytest.mark.asyncio
    async def test_person_filter_is_everyone_without_company_data(self):
        agent = self._agent_with(self._mixed_orders())

        result = await agent._dispatch("get_buyers", {"buyer_type": "person"})

        assert "Kawa i Spółka" not in result
        assert "Marek Zieliński" in result and "Katarzyna Wójcik" in result

    @pytest.mark.asyncio
    async def test_issued_filter_keeps_only_orders_with_an_invoice_attached(self):
        agent = self._agent_with(self._mixed_orders(), issued={"c2"})

        result = await agent._dispatch(
            "get_buyers", {"buyer_type": "company", "invoice_status": "issued"}
        )

        # Only the invoiced order counts toward the row — the company's other
        # order in the period has no invoice, so neither its value nor its
        # count may leak into an "invoices I issued" report.
        assert "| Kawa i Spółka | Firma | 7792445588 | `anna.firma` | 1 | 600,00 PLN | 1 | 01.05.2026 |" in result
        assert "Faktury VAT" in result

    @pytest.mark.asyncio
    async def test_missing_filter_keeps_only_requested_but_not_issued(self):
        agent = self._agent_with(self._mixed_orders(), issued={"c1", "c2"})

        result = await agent._dispatch("get_buyers", {"invoice_status": "missing"})

        assert "Marek Zieliński" in result
        assert "Kawa i Spółka" not in result
        # kasia never asked for an invoice, so she is not owed one.
        assert "`kasia`" not in result

    @pytest.mark.asyncio
    async def test_no_invoice_filter_spends_no_invoice_lookups(self):
        """One API call per order — a question that never mentioned invoices
        must not pay for them."""
        agent = self._agent_with(self._mixed_orders())

        await agent._dispatch("get_buyers", {})

        agent._allegro.invoices_issued_map.assert_not_called()

    @pytest.mark.asyncio
    async def test_period_defaults_to_the_current_year(self):
        from datetime import date

        agent = self._agent_with(self._mixed_orders())

        result = await agent._dispatch("get_buyers", {})

        date_from, date_to = agent._allegro.get_all_paid_orders_in_period.call_args.args
        year = date.today().year
        assert date_from.startswith(str(year - 1)) or date_from.startswith(str(year))
        assert f"{year}-01-01 – {date.today().isoformat()}" in result
        assert date_to >= date_from

    @pytest.mark.asyncio
    async def test_count_only_answers_in_one_sentence_without_a_table(self):
        agent = self._agent_with(self._mixed_orders())

        result = await agent._dispatch("get_buyers", {"count_only": True})

        assert result.startswith("Miałeś **3** kupujących w okresie")
        assert "|" not in result

    @pytest.mark.asyncio
    async def test_empty_period_names_the_filter_that_was_applied(self):
        agent = self._agent_with([])

        result = await agent._dispatch("get_buyers", {"buyer_type": "company"})

        assert result.startswith("Brak kupujących (firmy) w okresie")

    @pytest.mark.asyncio
    async def test_sort_by_orders_and_recent(self):
        orders = self._mixed_orders() + [
            self._order("p3", "kasia", 20.0, paid_at="2026-06-01T10:00:00Z"),
            self._order("p4", "kasia", 20.0, paid_at="2026-07-01T10:00:00Z"),
        ]
        agent = self._agent_with(orders)

        by_orders = await agent._dispatch("get_buyers", {"sort_by": "orders"})
        by_recent = await agent._dispatch("get_buyers", {"sort_by": "recent"})

        def first_row(result):
            return [ln for ln in result.splitlines() if ln.startswith("| ") and "---" not in ln][1]

        assert first_row(by_orders).startswith("| Katarzyna Wójcik |")  # 3 orders, lowest value
        assert first_row(by_recent).startswith("| Katarzyna Wójcik |")  # bought most recently
        assert first_row(await agent._dispatch("get_buyers", {})).startswith("| Kawa i Spółka |")

    @pytest.mark.asyncio
    async def test_failed_invoice_lookup_is_reported_not_counted_as_missing(self):
        agent = self._agent_with(self._mixed_orders())
        agent._allegro.invoices_issued_map = AsyncMock(
            side_effect=lambda ids: {oid: (None if oid == "c1" else oid == "c2") for oid in ids}
        )

        result = await agent._dispatch(
            "get_buyers", {"buyer_type": "company", "invoice_status": "issued"}
        )

        assert "Statusu faktury nie udało się sprawdzić dla 1 zamówienia" in result
