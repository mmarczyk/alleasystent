# AllEasystent — Wymagania funkcjonalne

**Wersja:** 1.1
**Data aktualizacji:** 2026-06-16
**Status:** Odzwierciedla aktualnie zaimplementowane funkcjonalności

---

## Spis treści

1. [Cel produktu](#1-cel-produktu)
2. [Wymagania funkcjonalne](#2-wymagania-funkcjonalne)
3. [Wymagania niefunkcjonalne](#3-wymagania-niefunkcjonalne)
4. [Architektura](#4-architektura)
5. [Integracja z Allegro](#5-integracja-z-allegro)
6. [System agentów AI](#6-system-agentów-ai)
7. [Baza wiedzy (RAG)](#7-baza-wiedzy-rag)
8. [Kanały komunikacji](#8-kanały-komunikacji)
9. [Persystencja danych](#9-persystencja-danych)
10. [API — lista endpointów](#10-api--lista-endpointów)
11. [Konfiguracja](#11-konfiguracja)
12. [Deployment](#12-deployment)
13. [Bezpieczeństwo](#13-bezpieczeństwo)
14. [Ograniczenia i znane zachowania](#14-ograniczenia-i-znane-zachowania)

---

## 1. Cel produktu

AllEasystent to asystent AI dla właścicieli sklepów na Allegro. Umożliwia zarządzanie zamówieniami, ofertami, wiadomościami od kupujących i kontem sprzedawcy przez interfejs konwersacyjny — bez konieczności klikania w panelu Allegro.

**Główne grupy funkcji:**

| ID | Obszar | Opis |
|---|---|---|
| F-01 | Integracja Allegro | Zarządzanie zamówieniami, ofertami, wiadomościami, kontem |
| F-02 | AI Routing | Automatyczne kierowanie zapytań do właściwego agenta |
| F-03 | Baza wiedzy | RAG — odpowiedzi na pytania o polityki, FAQ, produkty |
| F-04 | Interfejs webowy | PWA z historią rozmów i trybem offline |
| F-05 | Facebook Messenger | Webhook — obsługa konwersacji z kupującymi |
| F-06 | Powiadomienia push | VAPID — alerty o nowych zamówieniach i wiadomościach |

---

## 2. Wymagania funkcjonalne

### F-01 Integracja z Allegro

#### F-01.1 Autoryzacja

- System inicjuje OAuth2 Device Flow po wpisaniu zapytania wymagającego danych Allegro
- Użytkownik autoryzuje dostęp jednorazowo przez stronę Allegro
- Tokeny są automatycznie odświeżane przed wygaśnięciem
- Stan autoryzacji jest dostępny przez endpoint `/allegro/auth/status`

#### F-01.2 Obsługiwane operacje

| Operacja | Parametry wejściowe | Wynik |
|---|---|---|
| Lista zamówień | status, login kupującego, czy wysłane, limit (max 50) | ID, kupujący, status, kwota, pozycje |
| Szczegóły zamówienia | order_id | Kupujący, pozycje, dostawa, status płatności |
| Lista ofert | filtr po nazwie, limit (max 50) | ID, nazwa, cena, stan magazynowy |
| Szczegóły oferty | offer_id | Pełne dane oferty (max 3000 znaków) |
| Zmiana ceny | offer_id, cena w PLN (> 0) | Potwierdzenie |
| Aktualizacja stanu magazynowego | offer_id, ilość (≥ 0) | Potwierdzenie |
| Lista wątków wiadomości | limit (max 50) | ID wątku, temat, status odczytania |
| Wysłanie wiadomości | thread_id, treść | ID wiadomości |
| Dane konta | — | Login, email, firma, data rejestracji |
| Rozliczenia | limit (max 50) | Data, typ opłaty, kwota w PLN |

### F-02 Routing zapytań

- Klasyfikacja intencji odbywa się dwuetapowo: reguły słów kluczowych (bez LLM) → Gemini (fallback)
- Obsługiwane intencje: `allegro_orders`, `allegro_offers`, `allegro_messaging`, `allegro_account`, `general_knowledge`, `chitchat`
- Zapytania wielojęzyczne obsługiwane (PL/EN); agent odpowiada w języku pytania

### F-03 Baza wiedzy (RAG)

- Indeksowanie z pliku tekstowego/Markdown, katalogu produktów JSON, listy FAQ, aktywnych ofert Allegro
- Retrieval przed każdym zapytaniem do agenta (best-effort — błąd nie blokuje odpowiedzi)
- Parametr `RAG_TOP_K` kontroluje liczbę zwracanych dokumentów (domyślnie: 5)

### F-04 Interfejs webowy (PWA)

- Wielowątkowa historia rozmów przechowywana w LocalStorage
- Renderowanie Markdown z podświetlaniem składni
- Kopiowanie wiadomości, regeneracja ostatniej odpowiedzi, eksport do `.txt`
- Tryb offline (Service Worker, cache-first)
- Instalowalna jako aplikacja (manifest PWA)

### F-05 Facebook Messenger

- Obsługa wiadomości tekstowych, załączników i postbacków (quick replies, przyciski)
- Walidacja HMAC-SHA256 każdego przychodzącego żądania
- Długie odpowiedzi dzielone na fragmenty (max 1900 znaków) na granicach akapitów

### F-06 Powiadomienia push

- Subskrypcja przez PWA (VAPID)
- Obsługiwane platformy: iOS 16.4+ (PWA), Android, desktop

---

## 3. Wymagania niefunkcjonalne

| ID | Wymaganie | Wartość |
|---|---|---|
| NF-01 | Język odpowiedzi | Polski lub angielski — zgodny z językiem pytania |
| NF-02 | Czas odpowiedzi | Zależny od Gemini API; brak twardego limitu po stronie backendu |
| NF-03 | Pętla tool-use | Max 10 iteracji na jedno zapytanie |
| NF-04 | Rozmiar odpowiedzi Allegro | Max 3000 znaków dla szczegółów oferty |
| NF-05 | Kontener Docker | Uruchamiany jako non-root (`appuser`, UID 1000) |
| NF-06 | Embeddingi | Model lokalny (offline, wielojęzyczny) — brak zależności od zewnętrznego API |

---

## 4. Architektura

```
┌─────────────────────────────────────────────────────┐
│  Kanały wejścia                                     │
│  ┌──────────────┐  ┌──────────────────────────────┐ │
│  │  PWA (web/)  │  │  Facebook Messenger webhook  │ │
│  └──────┬───────┘  └──────────────┬───────────────┘ │
└─────────┼────────────────────────┼─────────────────┘
          │ POST /query            │ POST /webhook/facebook
          ▼                        ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI (main.py)                                  │
│  ┌──────────────────────────────────────────────┐  │
│  │  Orchestrator                                │  │
│  │  1. Ładuje historię (Firestore / in-memory)  │  │
│  │  2. RAG retrieve (best-effort)               │  │
│  │  3. Klasyfikacja intencji                    │  │
│  │  4. Routing do agenta                        │  │
│  │  5. Zapisuje historię                        │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
          │
   ┌──────┴──────────────┐
   ▼                     ▼
AllegroAgent          RAGAgent
(Gemini 2.0 Flash)    (Gemini 2.5 Flash)
   │
AllegroService → Allegro REST API
```

**Przepływ wiadomości:**

1. Wiadomość trafia przez `/query` lub webhook Messengera
2. Orkiestrator ładuje historię rozmowy z Firestore (fallback: pamięć)
3. RAG retriever pobiera kontekst z bazy wiedzy
4. Klasyfikacja intencji: reguły słów kluczowych → Gemini
5. Routing do odpowiedniego agenta
6. Agent wykonuje pętlę tool-use z Gemini (max 10 iteracji)
7. Odpowiedź wraca do kanału wyjściowego
8. Historia rozmowy zapisywana w Firestore

---

## 5. Integracja z Allegro

### OAuth2 Device Flow

1. Backend wysyła żądanie device code do `POST /auth/oauth/device`
2. Użytkownik otwiera link (`verification_uri_complete`) i zatwierdza dostęp
3. Backend odpytuje endpoint tokenowy co `interval` sekund
4. Po zatwierdzeniu tokeny zapisywane są do pliku `.allegro_tokens.json` lub GCP Secret Manager
5. Wygasłe tokeny są automatycznie odświeżane

**Zakresy (scopes):** `allegro:api:sale:offers:read`, `allegro:api:orders:read`, `allegro:api:orders:write`, `allegro:api:messaging`

---

## 6. System agentów AI

### Orkiestrator

**Etap 1 — reguły słów kluczowych (bez wywołania LLM):**

| Intencja | Słowa kluczowe (PL/EN) |
|---|---|
| `allegro_orders` | zamówien, paczk, dostaw, śledzeni, zwrot, reklamacj, faktur, order, tracking, shipment |
| `allegro_offers` | ofert, cen, produkt, stan magaz, aktywn, wystawion, listing, price, stock |
| `allegro_messaging` | wiadomoś, napisz do, kupując, message, buyer, odpowiedz |
| `allegro_account` | konto, opłat, prowizj, statystyk, rozliczen, account, fees, billing |
| `chitchat` | funkcj, możliwości, co potrafisz, cześć, hej, witaj, hello, hi |

**Etap 2 — Gemini (`gemini-2.0-flash`, max 30 tokenów):**

Dla zapytań niejednoznacznych, nierozpoznanych przez reguły. Fallback przy nieznanej intencji: `chitchat`.

### AllegroAgent

- Model: `gemini-2.0-flash`
- Sprawdza tokeny Allegro przed każdym zapytaniem; inicjuje device flow jeśli brak
- Zawsze pobiera świeże dane przez narzędzia (nie opiera się na wiedzy modelu)

### RAGAgent

- Model: `gemini-2.5-flash`
- Dwie role: agent dla zapytań `general_knowledge` + dostawca kontekstu dla pozostałych agentów
- Narzędzie: `search_knowledge_base` — przeszukuje ChromaDB

### BaseAgent — wspólne zasady

- Reguła języka (najwyższy priorytet w promptcie): odpowiadaj w języku pytania użytkownika, ignorując język wyników narzędzi
- Wyniki narzędzi w formacie OpenAI (`role: tool`, `tool_call_id`)

---

## 7. Baza wiedzy (RAG)

### Backendy

| Backend | Opis | Kiedy używać |
|---|---|---|
| ChromaDB (domyślny) | Lokalny SQLite, model embeddingów pobierany do kontenera | Development, Railway |
| Vertex AI | Zarządzany index GCP | Produkcja na GCP |

### Parametry ChromaDB

- Model embeddingów: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Metryka: cosine similarity
- Kolekcja: `store_knowledge`
- Chunking: 800 znaków z nakładką 100 znaków

### Źródła indeksowania

| Źródło | Endpoint | Format żądania |
|---|---|---|
| Plik tekstowy / Markdown | `POST /admin/rag/index-file` | `{"path": "...", "file_type": "text"}` |
| Katalog produktów JSON | `POST /admin/rag/index-file` | `{"path": "...", "file_type": "json_catalog"}` |
| FAQ | `POST /admin/rag/index-faq` | `{"items": [{"question": "...", "answer": "..."}]}` |
| Aktywne oferty Allegro | `POST /admin/rag/index-allegro-offers` | — |

---

## 8. Kanały komunikacji

### Interfejs webowy (PWA)

- Ekran powitalny z 4 sugestiami pytań
- Wielowątkowe rozmowy z historią w LocalStorage
- Automatyczne tytuły rozmów (pierwsze 50 znaków pytania)
- Renderowanie Markdown (Marked.js + Highlight.js)
- Kopiowanie wiadomości, regeneracja odpowiedzi, eksport do `.txt`
- Skróty: `Enter` — wyślij, `Shift+Enter` — nowa linia
- Service Worker v3 — cache-first, manifest PWA z ikonami SVG

### Facebook Messenger

- Weryfikacja webhooka: `GET /webhook/facebook` (zwraca `hub.challenge`)
- Obsługiwane zdarzenia: wiadomości tekstowe, załączniki, postbacki
- Ignorowane: `message_echoes`, `message_reads`, `message_reactions`
- Bezpieczeństwo: HMAC-SHA256 (`X-Hub-Signature-256`)
- Długie odpowiedzi: dzielone na fragmenty ≤ 1900 znaków na granicach akapitów/zdań

---

## 9. Persystencja danych

| Dane | Backend produkcja | Backend development | Fallback |
|---|---|---|---|
| Historia rozmów | Firestore (`conversations`) | Firestore | In-memory dict |
| Historia rozmów (frontend) | LocalStorage | LocalStorage | — |
| Tokeny Allegro | GCP Secret Manager | `.allegro_tokens.json` | — |
| Wektory RAG | Vertex AI | ChromaDB (SQLite) | — |
| Sesje / cache tokenów | Redis | Redis | — |

---

## 10. API — lista endpointów

| Metoda | Ścieżka | Opis |
|---|---|---|
| `GET` | `/health` | Health check — `{status, env}` |
| `GET` | `/` | Interfejs webowy (PWA) |
| `POST` | `/query` | Zapytanie do orkiestratora |
| `GET` | `/allegro/auth` | Inicjuje device flow, redirect do Allegro |
| `GET` | `/allegro/auth/status` | Stan autoryzacji: `idle \| pending \| authorized \| expired \| error` |
| `GET` | `/webhook/facebook` | Weryfikacja webhooku Messengera |
| `POST` | `/webhook/facebook` | Zdarzenia Messengera |
| `POST` | `/admin/rag/index-file` | Indeksowanie pliku |
| `POST` | `/admin/rag/index-faq` | Indeksowanie FAQ |
| `POST` | `/admin/rag/index-allegro-offers` | Indeksowanie ofert z Allegro |
| `POST` | `/admin/rag/query` | Test retrieval (debug) |

---

## 11. Konfiguracja

### Wymagane

| Zmienna | Opis |
|---|---|
| `GOOGLE_API_KEY` | Klucz Google AI Studio (Gemini) |

### Allegro

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `ALLEGRO_CLIENT_ID` | (wbudowany) | Client ID aplikacji Allegro |
| `ALLEGRO_CLIENT_SECRET` | (wbudowany) | Client Secret |
| `ALLEGRO_TOKEN_STORE` | `file` | `file` lub `secret_manager` |
| `ALLEGRO_TOKEN_FILE` | `.allegro_tokens.json` | Ścieżka do pliku tokenów |

### Facebook Messenger

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `FACEBOOK_PAGE_ACCESS_TOKEN` | `""` | Token strony Facebook |
| `FACEBOOK_VERIFY_TOKEN` | `alleasystent_verify_token` | Token weryfikacji webhooku |
| `FACEBOOK_APP_SECRET` | `""` | Sekret aplikacji (walidacja podpisu) |

### Modele AI

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model do złożonego rozumowania (RAG) |
| `GEMINI_MODEL_FAST` | `gemini-2.0-flash` | Model szybki (klasyfikacja, Allegro) |
| `GEMINI_MAX_TOKENS` | `16000` | Limit tokenów odpowiedzi |

### RAG

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `RAG_BACKEND` | `chromadb` | `chromadb` lub `vertex_ai` |
| `CHROMADB_PATH` | `./data/chromadb` | Ścieżka do bazy wektorowej |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Model embeddingów |
| `RAG_TOP_K` | `5` | Liczba dokumentów w kontekście |

### GCP

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `GCP_PROJECT_ID` | `""` | ID projektu GCP (Firestore, Pub/Sub) |
| `GCP_REGION` | `europe-central2` | Region |

### Aplikacja

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `APP_ENV` | `development` | `development` lub `production` |
| `LOG_LEVEL` | `INFO` | Poziom logowania |
| `PORT` | `8080` | Port serwera |

---

## 12. Deployment

### Railway (zalecane)

- Wymagana tylko zmienna `GOOGLE_API_KEY`
- Railway wykrywa `Dockerfile` automatycznie
- Model embeddingów pobierany podczas budowania obrazu (nie przy starcie kontenera)

**Docker (szczegóły):**
- Python 3.12-slim, multi-stage build
- Cache HuggingFace: `/app/.cache/huggingface` (`HF_HOME`)
- Użytkownik non-root: `appuser` (UID 1000)
- Port: 8080

### GCP Cloud Run

- Pipeline CI/CD: `cloudbuild.yaml`
- Artifact Registry: `europe-central2-docker.pkg.dev`
- Autoscaling: 0–10 instancji, 1 CPU, 2 GiB RAM, timeout 300s
- Sekrety przez GCP Secret Manager

### GitHub Pages (frontend-only)

- Wyzwalacz: push do `main`, zmiany w `web/**`
- Wdraża katalog `web/` na GitHub Pages
- Frontend łączy się z oddzielnym backendem przez ustawienie `backendUrl`

---

## 13. Bezpieczeństwo

| Obszar | Mechanizm |
|---|---|
| Webhook Facebook | Walidacja HMAC-SHA256 każdego żądania |
| Kontener Docker | Uruchamiany jako non-root (`appuser`) |
| Tokeny Allegro | Plik lokalny (dev) lub GCP Secret Manager (prod) |
| CORS | `allow_origins=["*"]` — **wymaga zawężenia na produkcji** |
| Allegro credentials | **Do przeniesienia z kodu do zmiennych środowiskowych** |

---

## 14. Ograniczenia i znane zachowania

| ID | Ograniczenie |
|---|---|
| L-01 | RAG jest pusty po starcie — baza wiedzy wymaga ręcznego zaindeksowania przez `/admin/rag/*` |
| L-02 | ChromaDB traci dane przy restarcie kontenera — brak persistent volume na Railway |
| L-03 | System obsługuje jedno konto Allegro na instancję |
| L-04 | Historia rozmów w Firestore (backend) i LocalStorage (frontend) są niezależne |
| L-05 | `allegro_token_store = "secret_manager"` jest w konfiguracji, ale implementacja nie jest ukończona |
| L-06 | Pub/Sub jest zintegrowany w kodzie, ale nieużywany aktywnie w głównym przepływie |
