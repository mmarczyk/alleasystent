"""Unit tests for the query-performance-by-phase part of analytics_service.py."""
from __future__ import annotations

import json

import pytest

from services import analytics_service as svc


class TestIntentLabel:
    """The dashboard's query-type label. Since the routing source collapsed to a
    bare "allegro", the tool that actually ran is what tells one store query
    from another — see _intent_label's docstring."""

    def test_tool_wins_over_the_source_half(self):
        assert svc._intent_label("allegro:table", "get_new_orders") == "Nowe zamówienia"
        # The wording says "zamówienia", the tool says billing — the tool is right.
        assert svc._intent_label("allegro:chat", "get_billing_summary") == "Rozliczenia"

    def test_unknown_tool_falls_back_to_humanized_name(self):
        assert svc._intent_label("allegro:chat", "some_future_tool") == "Some future tool"

    def test_no_tool_labels_off_the_intent(self):
        assert svc._intent_label("none:chat") == "Chitchat / inne"
        assert svc._intent_label("rag:document") == "Baza wiedzy [dokument]"
        assert svc._intent_label("allegro:table") == "Allegro [tabela]"

    def test_records_written_before_the_collapse_still_read(self):
        """Both Redis lists are capped ring buffers, not wiped on deploy, so the
        four old Allegro labels must stay readable until they age out."""
        assert svc._intent_label("allegro_orders:table") == "Zamówienia [tabela]"
        assert svc._intent_label("allegro_account:chat") == "Konto"
        assert svc._intent_label("chitchat") == "Chitchat / inne"


class TestLabelForPerf:
    def test_prefers_specific_tool_over_data_source(self):
        assert svc.label_for_perf("allegro", ["get_new_orders"]) == "Nowe zamówienia"
        assert svc.label_for_perf("allegro", ["get_orders"]) == "Zamówienia"

    def test_distinguishes_invoice_tools_from_generic_orders(self):
        assert svc.label_for_perf("allegro", ["get_orders_pending_invoice"]) == "Faktury do wystawienia"
        assert svc.label_for_perf("allegro", ["issue_invoice_for_order"]) == "Wystawianie faktury"

    def test_unknown_tool_falls_back_to_humanized_name(self):
        assert svc.label_for_perf("allegro", ["some_future_tool"]) == "Some future tool"

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


class TestRedact:
    """Redaction runs on real seller traffic before it leaves Redis, so a miss
    here is a data leak, not a cosmetic bug. See the note above the patterns
    in analytics_service for why values are REPLACED rather than removed."""

    def test_phone_written_the_way_a_human_types_it(self):
        """The case a naive "\\d{5,}" run-length rule silently publishes:
        "880 197 834" is three 3-digit groups, every one of them under any
        sane threshold, yet together an entire phone number."""
        assert svc.redact("Kto to jest 880 197 834?") == "Kto to jest <NUMER>?"
        assert svc.redact("Kto to jest 601 220 118?") == "Kto to jest <NUMER>?"
        assert svc.redact("klient z nr +48 880-197-834") == "klient z nr <NUMER>"
        assert svc.redact("czy mam klienta 880197834") == "czy mam klienta <NUMER>"

    def test_two_different_phones_redact_to_the_same_query(self):
        """The whole premise of redacting for an INTENT corpus: these are the
        same question, and the model should see them as one."""
        assert svc.redact("Kto to jest 601 220 118?") == svc.redact("Kto to jest 880 197 834?")

    def test_email_and_uuid(self):
        assert svc.redact("napisz do jan.kowalski@example.com") == "napisz do <EMAIL>"
        assert svc.redact(
            "szczegóły 3fa85f64-5717-4562-b3fc-2c963f66afa6"
        ) == "szczegóły <ID>"

    def test_dates_keep_their_own_placeholder(self):
        """A date is not personal data and it IS the intent — a period question
        must stay recognisable as one after redaction."""
        assert svc.redact("koszty od 2026-09-01 do 2026-09-09") == "koszty od <DATA> do <DATA>"
        assert svc.redact("zamówienia z 15.03") == "zamówienia z <DATA>"

    def test_short_numbers_survive(self):
        """"ostatnie 3 zamówienia" carries a limit the classifier should see;
        only long, identifier-shaped runs are personal data."""
        assert svc.redact("pokaż ostatnie 3 zamówienia") == "pokaż ostatnie 3 zamówienia"
        assert svc.redact("top 10 ofert") == "top 10 ofert"

    def test_buyer_login(self):
        assert svc.redact("z konta np1988 kupował") == "z konta <LOGIN> kupował"
        assert svc.redact("napisz do jan_kowalski88") == "napisz do <LOGIN>"

    def test_login_is_redacted_whole_not_just_its_digit_half(self):
        """The shape rule anchored on the digit-bearing half and left "anna." —
        a first name — standing in the clear."""
        assert svc.redact("od użytkownika anna.kowalska88") == "od użytkownika <LOGIN>"

    def test_login_without_a_digit_needs_the_context_anchored_rule(self):
        """"sklep-abc" carries neither digit nor underscore, so the shape rule
        cannot see it without swallowing every hyphenated word in the language.
        The account word in front is what makes it identifiable."""
        assert svc.redact("login: sklep-abc") == "login: <LOGIN>"

    def test_postcode(self):
        assert svc.redact("wyślij na 00-950 Warszawa") == "wyślij na <NUMER> Warszawa"

    @pytest.mark.parametrize("query", [
        "ile mam nowych zamówień?",
        "pokaż ostatnie 3 zamówienia",
        "co muszę wysłać dzisiaj",
        "jaka jest polityka zwrotów?",
        "top 10 ofert",
        "podsumuj sprzedaż z tego roku z podziałem na miesiące",
        "czy mam wiadomości od kupujących",
        "zmień cenę oferty na 49,99 zł",
        "Ile paczek muszę dziś nadać?",
        "włącz powiadomienia o zwrotach",
    ])
    def test_ordinary_queries_survive_untouched(self, query):
        """Redaction leans on recall, so the guard against it eating the corpus
        is this list: a plain seller question must come out byte-identical."""
        assert svc.redact(query) == query


class TestExportTrainingCorpus:
    def _rows(self):
        return [
            {"text": "nowe zamówienia", "intent": "allegro:table",
             "tool": "get_new_orders", "path": "keyword", "ts": 1},
            {"text": "Kto to jest 880 197 834?", "intent": "allegro:chat",
             "tool": "find_buyer_by_contact", "path": "llm", "ts": 2},
            {"text": "sprawdź jeszcze raz", "intent": "allegro:table",
             "tool": "get_new_orders", "path": "inherited", "ts": 3},
            {"text": "co u ciebie", "intent": "none:chat", "path": "llm", "ts": 4},
            # Written before the field existed.
            {"text": "stare zapytanie", "intent": "allegro_orders:table", "ts": 5},
        ]

    @pytest.mark.asyncio
    async def test_only_the_llm_branches_are_exported(self, monkeypatch):
        """The keyword branch reproduces a matcher that still runs in front of
        any model, so learning from it buys nothing and skews the distribution
        — see _TRAINABLE_PATHS."""
        async def fake_fetch():
            return self._rows(), []
        monkeypatch.setattr(svc, "_fetch_all", fake_fetch)

        rows = await svc.export_training_corpus()

        assert [r["path"] for r in rows] == ["llm", "inherited", "llm"]
        assert "nowe zamówienia" not in [r["text"] for r in rows]
        assert "stare zapytanie" not in [r["text"] for r in rows]

    @pytest.mark.asyncio
    async def test_exported_text_is_redacted(self, monkeypatch):
        async def fake_fetch():
            return self._rows(), []
        monkeypatch.setattr(svc, "_fetch_all", fake_fetch)

        rows = await svc.export_training_corpus()

        assert rows[0]["text"] == "Kto to jest <NUMER>?"
        assert not any("880" in r["text"] for r in rows)

    @pytest.mark.asyncio
    async def test_label_is_the_source_half_only(self, monkeypatch):
        """The format half is decided downstream by the tool — predicting it is
        not this classifier's job."""
        async def fake_fetch():
            return self._rows(), []
        monkeypatch.setattr(svc, "_fetch_all", fake_fetch)

        rows = await svc.export_training_corpus()

        assert [r["source"] for r in rows] == ["allegro", "allegro", "none"]

    @pytest.mark.asyncio
    async def test_no_user_id_reaches_the_corpus(self, monkeypatch):
        async def fake_fetch():
            return [{"text": "co u ciebie", "intent": "none:chat",
                     "path": "llm", "uid": "google-sub-12345", "ts": 1}], []
        monkeypatch.setattr(svc, "_fetch_all", fake_fetch)

        rows = await svc.export_training_corpus()

        assert "uid" not in rows[0]
        assert "google-sub-12345" not in json.dumps(rows)

    @pytest.mark.asyncio
    async def test_empty_paths_filter_exports_everything(self, monkeypatch):
        async def fake_fetch():
            return self._rows(), []
        monkeypatch.setattr(svc, "_fetch_all", fake_fetch)

        rows = await svc.export_training_corpus(paths=())

        assert len(rows) == 5
        assert rows[-1]["path"] == "unknown"


class TestPathSplitInStats:
    @pytest.mark.asyncio
    async def test_shares_are_reported_and_old_records_count_as_unknown(self, monkeypatch):
        """The LLM share is the number that decides whether a local classifier
        is worth building, so records predating the field must be visible as
        "unknown" rather than silently inflating one of the real branches."""
        async def fake_fetch():
            return [
                {"text": "a", "intent": "allegro:chat", "path": "keyword"},
                {"text": "b", "intent": "allegro:chat", "path": "keyword"},
                {"text": "c", "intent": "none:chat", "path": "llm"},
                {"text": "d", "intent": "allegro:chat"},
            ], []
        monkeypatch.setattr(svc, "_fetch_all", fake_fetch)

        stats = await svc.get_stats()

        assert {p["path"]: p["count"] for p in stats["paths"]} == {
            "keyword": 2, "llm": 1, "unknown": 1,
        }
        assert {p["path"]: p["pct"] for p in stats["paths"]}["llm"] == 25
