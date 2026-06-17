# Testy API — AllEasystent

Scenariusze testowe weryfikujące wymagania z PRD przez wywołania `POST /query`.

## Moduły

| Plik | Zakres |
|---|---|
| `test_01_routing.py` | Klasyfikacja intencji — keyword routing i LLM routing |
| `test_02_chitchat.py` | Powitania i pytania o możliwości asystenta |
| `test_03_allegro_auth.py` | OAuth2 browser flow (/allegro/login), zachowanie bez/z autoryzacją, /auth/me |
| `test_04_orders.py` | Zamówienia — listowanie, szczegóły, filtry, polskie tłumaczenia statusów |
| `test_05_offers.py` | Oferty — listowanie, zmiana ceny, stan magazynowy |
| `test_06_messaging.py` | Wiadomości od kupujących — lista (pola, status odczytania), wysyłanie (potwierdzenie, długa treść), EN |
| `test_07_account.py` | Konto sprzedawcy (pola, data rejestracji, EN) i rozliczenia (pola, sortowanie, limit, EN) |
| `test_08_language.py` | Wielojęzyczność (PL/EN) |
| `test_09_rag.py` | Baza wiedzy — routing, pusta baza, indeksowanie |
| `test_10_api_and_health.py` | Health check, kształt API, edge cases, endpointy push (/push/pending, /push/status) |
| `test_11_conversations.py` | Kontekst wieloturowy i izolacja sesji |

## Uruchomienie

### Wymagania

```bash
pip install pytest httpx
```

### Testy bez autoryzacji Allegro (większość)

```bash
# Domyślnie testuje http://localhost:8080
cd tests
pytest -v

# Lub wskaż URL działającego backendu
ALLEASYSTENT_URL=https://twoja-aplikacja.railway.app pytest -v
```

### Testy z autoryzacją Allegro

1. Otwórz `https://twoja-aplikacja.railway.app/allegro/auth` w przeglądarce
2. Zatwierdź dostęp na stronie Allegro
3. Sprawdź status: `GET /allegro/auth/status` → `{"status": "authorized"}`
4. Uruchom testy z flagą:

```bash
ALLEASYSTENT_URL=https://twoja-aplikacja.railway.app ALLEGRO_AUTHED=1 pytest -v
```

### Uruchomienie wybranego modułu

```bash
pytest test_01_routing.py -v
pytest test_08_language.py -v
```

### Uruchomienie tylko testów bez autoryzacji

```bash
pytest -v -m "not skipif"
# lub po prostu pomiń ALLEGRO_AUTHED=1 — testy @requires_allegro zostaną pominięte automatycznie
```

## Konwencje

- **Asercje na wzorcach, nie na dokładnym tekście** — odpowiedzi LLM są niedeterministyczne
- **Izolacja sesji** — każdy test generuje unikalny `session_id` przez `new_session()`
- **`@requires_allegro`** — testy oznaczone tym dekoratorem są pomijane gdy `ALLEGRO_AUTHED != 1`
- **Timeout 60s** — zapytania do LLM mogą trwać kilkanaście sekund
