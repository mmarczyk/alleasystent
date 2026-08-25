"""Unit tests for the query-performance-by-phase part of analytics_service.py."""
from __future__ import annotations

import pytest

from services import analytics_service as svc


class TestLabelForPerf:
    def test_prefers_specific_tool_over_data_source(self):
        assert svc.label_for_perf("allegro_orders", ["get_new_orders"]) == "Nowe zamówienia"
        assert svc.label_for_perf("allegro_orders", ["get_orders"]) == "Zamówienia"

    def test_distinguishes_invoice_tools_from_generic_orders(self):
        assert svc.label_for_perf("allegro_orders", ["get_orders_pending_invoice"]) == "Faktury do wystawienia"
        assert svc.label_for_perf("allegro_orders", ["issue_invoice_for_order"]) == "Wystawianie faktury"

    def test_unknown_tool_falls_back_to_humanized_name(self):
        assert svc.label_for_perf("allegro_orders", ["some_future_tool"]) == "Some future tool"

    def test_no_tools_falls_back_to_data_source_label(self):
        assert svc.label_for_perf("rag", None) == "Baza wiedzy"
        assert svc.label_for_perf("none", []) == "Chitchat / inne"

    def test_unknown_data_source_falls_back_to_raw_value(self):
        assert svc.label_for_perf("some_new_source", None) == "some_new_source"


class TestGetPerfStats:
    @pytest.mark.asyncio
    async def test_empty_when_no_data(self, monkeypatch):
        async def fake_fetch():
            return []
        monkeypatch.setattr(svc, "_fetch_perf", fake_fetch)

        result = await svc.get_perf_stats()
        assert result["series"] == []
        assert result["phase_keys"] == svc._PHASE_ORDER
        assert result["phase_labels"] == [svc._PHASE_LABELS[p] for p in svc._PHASE_ORDER]

    @pytest.mark.asyncio
    async def test_averages_per_label_and_collapses_tool_stages(self, monkeypatch):
        async def fake_fetch():
            return [
                {
                    "label": "Nowe zamówienia", "total_ms": 1000,
                    "phases": {"classify": 800, "tool:get_new_orders": 200},
                },
                {
                    "label": "Nowe zamówienia", "total_ms": 2000,
                    "phases": {"classify": 1600, "tool:get_new_orders": 400},
                },
                {
                    "label": "Zamówienia", "total_ms": 500,
                    "phases": {"classify": 100, "tool:get_orders": 400},
                },
            ]
        monkeypatch.setattr(svc, "_fetch_perf", fake_fetch)

        result = await svc.get_perf_stats()
        by_label = {s["label"]: s for s in result["series"]}

        new_orders = by_label["Nowe zamówienia"]
        assert new_orders["count"] == 2
        assert new_orders["avg_total_ms"] == 1500.0
        assert new_orders["phases"]["classify"] == 1200.0
        assert new_orders["phases"]["allegro_call"] == 300.0
        # "tool:get_new_orders" must not survive as its own key
        assert "tool:get_new_orders" not in new_orders["phases"]

        assert by_label["Zamówienia"]["count"] == 1

    @pytest.mark.asyncio
    async def test_series_sorted_by_count_descending(self, monkeypatch):
        async def fake_fetch():
            return [
                {"label": "Rzadkie", "total_ms": 100, "phases": {}},
                {"label": "Częste", "total_ms": 100, "phases": {}},
                {"label": "Częste", "total_ms": 100, "phases": {}},
            ]
        monkeypatch.setattr(svc, "_fetch_perf", fake_fetch)

        result = await svc.get_perf_stats()
        assert [s["label"] for s in result["series"]] == ["Częste", "Rzadkie"]
