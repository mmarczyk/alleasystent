# Matryca testów narzędzi agenta Allegro

Uruchamia **każde** narzędzie z `agents/allegro/allegro_tools.py` na syntetycznych
danych sklepu i pokazuje, co narzędzie zwraca — w konsoli, w testach i jako
zrzuty ekranu z prawdziwego interfejsu PWA.

```
tests/tools/
├── dataset.py             dane sklepu w surowym formacie API Allegro
├── fake_allegro.py        atrapy REST-owe Allegro i inFakt (httpx MockTransport)
├── harness.py             prawdziwy AllegroAgent podpięty do atrap
├── cases.py               matryca przypadków (1+ na każde narzędzie)
├── runner.py              uruchamia matrycę, wypisuje/eksportuje wyniki
├── test_all_tools.py      testy pytest (pokrycie + wynik każdego narzędzia)
└── capture_screenshots.py replay wyników w PWA + galeria zrzutów
```

## Co jest prawdziwe, a co udawane

Prawdziwe: `AllegroAgent`, `AllegroService`, `InfaktService`, wszystkie
formattery, paginacja, cache TTL, filtrowanie po stronie klienta, front-end PWA.

Udawane: warstwa HTTP (`httpx.MockTransport` zamiast api.allegro.pl / infakt.pl)
oraz flagi monitoringu (normalnie w Redis).

Pominięte: krok LLM. W produkcji tekst narzędzia trafia jeszcze do Gemini, który
układa z niego finalną odpowiedź (tabela / dashboard / dokument). Zrzuty
pokazują więc **surowe dane narzędzia** wyrenderowane przez prawdziwy front-end.

## Użycie

```bash
pip install -r tests/unit/requirements-unit.txt      # + playwright, pillow do zrzutów

# 1. Testy (pytest)
pytest tests/tools/

# 2. Podsumowanie w konsoli + eksport JSON
python -m tests.tools.runner --json artifacts/tool-results.json
python -m tests.tools.runner --only get_new_orders get_sales_summary

# 3. Zrzuty ekranu + galeria HTML
python -m tests.tools.capture_screenshots --out artifacts/tool-screenshots --standalone
```

Zrzuty wymagają Chromium dla Playwrighta (`playwright install chromium`) oraz
plików vendor front-endu (`web/js/vendor/`, `web/css/vendor/`) — te same, które
`.github/workflows/deploy-chat.yml` pobiera przy deployu. Bez nich strona nadal
działa (markdown renderowany fallbackiem), tylko bez kolorowania składni.

## Dodanie nowego narzędzia

`test_every_tool_has_a_case` pilnuje, żeby każde narzędzie z `ALLEGRO_TOOLS`
miało przypadek w `cases.py`. Nowe narzędzie = nowy `Case` (i, jeśli sięga po
nowy endpoint, obsługa tej ścieżki w `fake_allegro.py`).
