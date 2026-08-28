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

    @pytest.mark.asyncio
    async def test_hours_filters_out_older_entries(self, monkeypatch):
        import time as time_mod
        now = time_mod.time()
        monkeypatch.setattr(svc.time, "time", lambda: now)

        async def fake_fetch():
            return [
                {"label": "Nowe zamówienia", "total_ms": 1000, "ts": now - 3600, "phases": {}},   # 1h ago
                {"label": "Nowe zamówienia", "total_ms": 5000, "ts": now - 30 * 3600, "phases": {}},  # 30h ago
            ]
        monkeypatch.setattr(svc, "_fetch_perf", fake_fetch)

        result = await svc.get_perf_stats(hours=24)
        by_label = {s["label"]: s for s in result["series"]}
        assert by_label["Nowe zamówienia"]["count"] == 1
        assert by_label["Nowe zamówienia"]["avg_total_ms"] == 1000.0

    @pytest.mark.asyncio
    async def test_hours_none_keeps_full_history(self, monkeypatch):
        async def fake_fetch():
            return [
                {"label": "Nowe zamówienia", "total_ms": 1000, "ts": 1, "phases": {}},
                {"label": "Nowe zamówienia", "total_ms": 5000, "ts": 2, "phases": {}},
            ]
        monkeypatch.setattr(svc, "_fetch_perf", fake_fetch)

        result = await svc.get_perf_stats()
        assert result["series"][0]["count"] == 2


class TestColdStartStats:
    """See services/analytics_service._cold_start_stats and
    agents/orchestrator.py._mark_request — a Cloud Run cold start
    (--min-instances=0) happens entirely before any StageTimer starts, so
    it's otherwise invisible in the phase breakdown."""

    @pytest.mark.asyncio
    async def test_splits_cold_and_warm_averages(self, monkeypatch):
        async def fake_fetch():
            return [
                {"label": "Nowe zamówienia", "total_ms": 12000, "phases": {}, "cold": True},
                {"label": "Nowe zamówienia", "total_ms": 2000, "phases": {}, "cold": False},
                {"label": "Nowe zamówienia", "total_ms": 3000, "phases": {}, "cold": False},
            ]
        monkeypatch.setattr(svc, "_fetch_perf", fake_fetch)

        result = await svc.get_perf_stats()
        cold_start = result["cold_start"]
        assert cold_start["cold_count"] == 1
        assert cold_start["warm_count"] == 2
        assert cold_start["cold_avg_total_ms"] == 12000.0
        assert cold_start["warm_avg_total_ms"] == 2500.0

    @pytest.mark.asyncio
    async def test_no_cold_entries_gives_none_average(self, monkeypatch):
        async def fake_fetch():
            return [{"label": "Nowe zamówienia", "total_ms": 2000, "phases": {}, "cold": False}]
        monkeypatch.setattr(svc, "_fetch_perf", fake_fetch)

        result = await svc.get_perf_stats()
        assert result["cold_start"]["cold_count"] == 0
        assert result["cold_start"]["cold_avg_total_ms"] is None

    @pytest.mark.asyncio
    async def test_missing_cold_field_treated_as_warm(self, monkeypatch):
        """Entries logged before this field existed have no 'cold' key at
        all — must not crash and must count as warm, not cold."""
        async def fake_fetch():
            return [{"label": "Nowe zamówienia", "total_ms": 2000, "phases": {}}]
        monkeypatch.setattr(svc, "_fetch_perf", fake_fetch)

        result = await svc.get_perf_stats()
        assert result["cold_start"]["cold_count"] == 0
        assert result["cold_start"]["warm_count"] == 1

    @pytest.mark.asyncio
    async def test_present_even_when_no_data(self, monkeypatch):
        async def fake_fetch():
            return []
        monkeypatch.setattr(svc, "_fetch_perf", fake_fetch)

        result = await svc.get_perf_stats()
        assert result["cold_start"] == {
            "cold_count": 0, "warm_count": 0,
            "cold_avg_total_ms": None, "warm_avg_total_ms": None,
        }
