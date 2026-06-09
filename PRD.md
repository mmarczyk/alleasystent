# AllEasystent — Product Requirements Document

**Wersja:** 1.0  
**Data:** 2026-06-09  
**Status:** Odzwierciedla aktualnie zaimplementowane funkcjonalności

---

## 1. Cel produktu

AllEasystent to asystent AI dla właścicieli sklepów na Allegro. Umożliwia zarządzanie zamówieniami, ofertami, wiadomościami od kupujących i kontem sprzedawcy przez interfejs konwersacyjny — bez konieczności klikania w panelu Allegro. System obsługuje wielokanałowy dostęp: interfejs webowy (PWA) oraz Facebook Messenger.

---

## 2. Architektura systemu

### 2.1 Warstwy

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

### 2.2 Przepływ wiadomości

1. Wiadomość trafia przez `/query` lub webhook Messengera
2. Orkiestrator ładuje historię rozmowy z Firestore (fallback: pamięć)
3. RAG retriever pobiera kontekst z bazy wiedzy (opcjonalnie)
4. Klasyfikacja intencji: najpierw reguły słów kluczowych, potem Gemini
5. Routing do odpowiedniego agenta
6. Agent wykonuje pętlę tool-use z Gemini
7. Odpowiedź wraca do kanału wyjściowego
8. Historia rozmowy zapisywana w Firestore

---

## 3. Integracja z Allegro

### 3.1 Autoryzacja — OAuth2 Device Flow

**Wymaganie:** Sprzedawca musi jednorazowo autoryzować dostęp do konta Allegro.

**Przepływ:**
1. Backend wysyła żądanie device code do `POST /auth/oauth/device`
2. Użytkownik otwiera link (`verification_uri_complete`) i zatwierdza dostęp na stronie Allegro
3. Backend odpytuje (`polling`) endpoint tokenowy co `interval` sekund w tle
4. Po zatwierdzeniu tokeny zapisywane są do pliku `.allegro_tokens.json`
5. Przy każdym żądaniu API sprawdzana jest ważność tokenu; wygasłe tokeny są automatycznie odświeżane

**Zakresy (scopes):**
- `allegro:api:sale:offers:read`
- `allegro:api:orders:read`
- `allegro:api:orders:write`
- `allegro:api:messaging`

**Endpointy pomocnicze:**
- `GET /allegro/auth` — inicjuje flow, przekierowuje przeglądarkę do Allegro
- `GET /allegro/auth/status` — zwraca stan: `idle | pending | authorized | expired | error`

**Przechowywanie tokenów:** plik `.allegro_tokens.json` (development) lub GCP Secret Manager (produkcja, konfigurowane przez `allegro_token_store`).

---

### 3.2 Obsługiwane operacje Allegro

| Operacja | Opis | Parametry | Zwraca |
|---|---|---|---|
| **Zamówienia** | | | |
| Pobierz listę zamówień | Ostatnie zamówienia z filtrowaniem | status, login kupującego, fulfillment_status, czy wysłane, limit (max 50) | ID, kupujący, status, kwota, pozycje |
| Pobierz szczegóły zamówienia | Kompletne dane zamówienia | order_id | Kupujący, pozycje, dostawa, status płatności |
| **Oferty** | | | |
| Pobierz aktywne oferty | Lista ofert sklepu | filtr po nazwie, limit (max 50) | ID, nazwa, cena, stan magazynowy |
| Pobierz szczegóły oferty | Pełne dane oferty | offer_id | Cały JSON oferty (max 3000 znaków) |
| Zmień cenę oferty | Aktualizacja ceny | offer_id, cena w PLN (>0) | Potwierdzenie |
| Zaktualizuj stan magazynowy | Zmiana dostępnej ilości | offer_id, ilość (≥0) | Potwierdzenie |
| **Wiadomości** | | | |
| Pobierz wątki wiadomości | Lista konwersacji z kupującymi | limit (max 50) | ID wątku, temat, status odczytania, czas ostatniej wiadomości |
| Wyślij wiadomość do kupującego | Odpowiedź w wątku | thread_id, treść | ID wiadomości |
| **Konto** | | | |
| Dane konta sprzedawcy | Informacje o koncie | — | Login, email, firma, data rejestracji |
| Podsumowanie rozliczeń | Ostatnie opłaty i prowizje | limit (max 50) | Data, typ opłaty, kwota w PLN |

---

## 4. System agentów

### 4.1 Orkiestrator

Centralny router odpowiedzialny za cały przepływ przetwarzania wiadomości.

**Klasyfikacja intencji — etap 1: reguły słów kluczowych (bez wywołania LLM):**

| Intencja | Słowa kluczowe (PL/EN) |
|---|---|
| `allegro_orders` | zamówien, paczk, dostaw, śledzeni, zwrot, reklamacj, faktur, order, tracking, shipment |
| `allegro_offers` | ofert, cen, produkt, stan magaz, aktywn, wystawion, listing, price, stock |
| `allegro_messaging` | wiadomoś, napisz do, kupując, message, buyer, odpowiedz |
| `allegro_account` | konto, opłat, prowizj, statystyk, rozliczen, account, fees, billing |
| `chitchat` | funkcj, możliwości, co potrafisz, co umiesz, cześć, hej, witaj, hello, hi |

**Klasyfikacja intencji — etap 2: Gemini (dla niejednoznacznych zapytań):**
- Model: `gemini-2.0-flash`
- Max tokens: 30
- Fallback przy nieznanej intencji: `chitchat`
- Parsowanie: exact match → substring match → fallback

**Routing:**
- `allegro_*` → AllegroAgent
- `general_knowledge` → RAGAgent
- `chitchat` → odpowiedź Gemini (bez narzędzi, z listą możliwości)

### 4.2 AllegroAgent

- Model: `gemini-2.0-flash`
- Przed każdym zapytaniem sprawdza tokeny Allegro; jeśli brak lub wygasłe — inicjuje device flow
- Pętla tool-use: max 10 iteracji
- Zawsze pobiera świeże dane przez narzędzia (nie opiera się na wiedzy modelu)
- Język odpowiedzi: polski lub angielski — zgodny z językiem pytania

### 4.3 RAGAgent

- Model: `gemini-2.5-flash`
- Dwie role: agent (dla `general_knowledge`) i dostarczyciel kontekstu (dla każdego agenta)
- Narzędzie: `search_knowledge_base` — przeszukuje ChromaDB pod kątem polityk, FAQ, produktów
- Jeśli baza jest pusta lub retrieval się nie powiedzie — odpowiada bez kontekstu

### 4.4 BaseAgent — wspólne zasady

- Prompt systemowy: opis agenta + kontekst RAG (jeśli dostępny)
- Reguła języka (zawsze na końcu promptu, najwyższy priorytet):
  > Jeśli użytkownik pisze po polsku — odpowiadaj wyłącznie po polsku. Jeśli po angielsku — po angielsku. Wyniki narzędzi mogą być po angielsku — zignoruj to i odpowiedz w języku użytkownika.
- Wyniki narzędzi: format OpenAI (`role: tool`, `tool_call_id`)

---

## 5. Baza wiedzy (RAG)

### 5.1 Backend ChromaDB (domyślny)

- Model embeddingów: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (lokalny, wielojęzyczny)
- Metryka: cosine similarity
- Kolekcja: `store_knowledge`
- Przechowywanie: plik SQLite w `chromadb_path` (domyślnie `./data/chromadb`)
- Wszystkie operacje wykonywane asynchronicznie w wątku (`asyncio.to_thread`) — nie blokują event loop

### 5.2 Backend Vertex AI (opcjonalny, GCP)

- Embeddingi: `textembedding-gecko-multilingual@001`
- Vector Search: zarządzany index endpoint

### 5.3 Źródła indeksowania

| Źródło | Endpoint | Format |
|---|---|---|
| Plik tekstowy / Markdown | `POST /admin/rag/index-file` | `{path, file_type: "text"}` |
| Katalog produktów JSON | `POST /admin/rag/index-file` | `{path, file_type: "json_catalog"}` |
| FAQ | `POST /admin/rag/index-faq` | `{items: [{question, answer}]}` |
| Aktywne oferty Allegro | `POST /admin/rag/index-allegro-offers` | — |

**Chunking:** 800 znaków z nakładką 100 znaków dla plików tekstowych.

### 5.4 Parametry retrieval

- `rag_top_k`: liczba zwracanych dokumentów (domyślnie: 5)
- Retrieval wywoływany przed każdym routingiem do agenta
- Błąd retrieval nie blokuje odpowiedzi (best-effort)

---

## 6. Kanały komunikacji

### 6.1 Interfejs webowy (PWA)

**Technologia:** HTML + CSS + Vanilla JS, bez frameworka

**Funkcjonalności:**
- Ekran powitalny z 4 sugestiami przykładowych pytań
- Wielowątkowe rozmowy z historią w LocalStorage
- Automatyczne tytuły rozmów (pierwsze 50 znaków pytania)
- Renderowanie Markdown z podświetlaniem składni (Marked.js + Highlight.js)
- Kopiowanie wiadomości do schowka
- Regeneracja ostatniej odpowiedzi asystenta
- Eksport rozmowy do pliku `.txt`
- Usuwanie całej historii

**Ustawienia:**
- Backend URL (opcjonalny; przy pustym polu używa relative `/query`)

**Skróty klawiszowe:**
- `Enter` — wyślij
- `Shift+Enter` — nowa linia

**PWA / Offline:**
- Service Worker v3 — cache-first dla zasobów statycznych
- Manifest: ikony 192×512 SVG, tryb `standalone`, motyw ciemny (`#0f0f1a`)
- Buforuje: HTML, CSS, JS, manifest

### 6.2 Facebook Messenger

**Weryfikacja webhooka:** `GET /webhook/facebook` — zwraca `hub.challenge` przy poprawnym `verify_token`

**Obsługiwane zdarzenia:**
- Wiadomości tekstowe
- Załączniki (tagowane jako `[Attachment: typ]`)
- Postback (przyciski / quick replies, tagowane jako `[Button: tytuł]`)

**Ignorowane zdarzenia:** `message_echoes`, `message_reads`, `message_reactions`

**Bezpieczeństwo:** walidacja podpisu HMAC-SHA256 (`X-Hub-Signature-256`)

**Wysyłanie odpowiedzi:**
1. Pokaż wskaźnik pisania (`typing_on`)
2. Podziel długie odpowiedzi (max 1900 znaków) na granicach akapitów/zdań
3. Wyślij każdy fragment przez Facebook Send API
4. Ukryj wskaźnik pisania (`typing_off`)

---

## 7. Persystencja danych

### 7.1 Historia rozmów — Firestore

- Kolekcja: `conversations` (konfigurowalna)
- Operacje: `get`, `save`, `get_or_create`, `list` (z filtrowaniem po kanale)
- Fallback: in-memory dict (gdy `gcp_project_id` nie ustawiony)

### 7.2 Historia rozmów — LocalStorage (frontend)

- Klucz: `ae_conversations`
- Format: `[{id, title, messages: [{role, content, ts}], createdAt}]`
- Session ID: `Date.now().toString()`

### 7.3 Tokeny Allegro

- Plik: `.allegro_tokens.json` (development)
- GCP Secret Manager (production, konfigurowane przez `allegro_token_store = "secret_manager"`)

---

## 8. API — pełna lista endpointów

| Metoda | Ścieżka | Opis |
|---|---|---|
| `GET` | `/health` | Health check — `{status, env}` |
| `GET` | `/allegro/auth` | Inicjuje device flow, redirect do Allegro |
| `GET` | `/allegro/auth/status` | Stan autoryzacji |
| `POST` | `/query` | Zapytanie bezpośrednie do orkiestratora |
| `GET` | `/webhook/facebook` | Weryfikacja webhooku Messengera |
| `POST` | `/webhook/facebook` | Zdarzenia Messengera |
| `POST` | `/admin/rag/index-file` | Indeksowanie pliku |
| `POST` | `/admin/rag/index-faq` | Indeksowanie FAQ |
| `POST` | `/admin/rag/index-allegro-offers` | Indeksowanie ofert z Allegro |
| `POST` | `/admin/rag/query` | Test retrieval |
| `GET` | `/` | Interfejs webowy (PWA) |

---

## 9. Konfiguracja

Wszystkie parametry odczytywane z pliku `.env` lub zmiennych środowiskowych.

### Wymagane

| Zmienna | Opis |
|---|---|
| `GOOGLE_API_KEY` | Klucz Google AI Studio (Gemini) |

### Opcjonalne — Allegro

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `ALLEGRO_CLIENT_ID` | (wbudowany) | Client ID aplikacji Allegro |
| `ALLEGRO_CLIENT_SECRET` | (wbudowany) | Client Secret |
| `ALLEGRO_TOKEN_STORE` | `file` | `file` lub `secret_manager` |
| `ALLEGRO_TOKEN_FILE` | `.allegro_tokens.json` | Ścieżka do pliku tokenów |

### Opcjonalne — Facebook Messenger

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `FACEBOOK_PAGE_ACCESS_TOKEN` | `""` | Token strony Facebook |
| `FACEBOOK_VERIFY_TOKEN` | `alleasystent_verify_token` | Token weryfikacji webhooku |
| `FACEBOOK_APP_SECRET` | `""` | Sekret aplikacji (walidacja podpisu) |

### Opcjonalne — Modele AI

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model do złożonego rozumowania |
| `GEMINI_MODEL_FAST` | `gemini-2.0-flash` | Model szybki (klasyfikacja, Allegro) |
| `GEMINI_MAX_TOKENS` | `16000` | Limit tokenów odpowiedzi |

### Opcjonalne — RAG

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `RAG_BACKEND` | `chromadb` | `chromadb` lub `vertex_ai` |
| `CHROMADB_PATH` | `./data/chromadb` | Ścieżka do bazy wektorowej |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Model embeddingów |
| `RAG_TOP_K` | `5` | Liczba dokumentów w kontekście |

### Opcjonalne — GCP

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `GCP_PROJECT_ID` | `""` | ID projektu GCP (Firestore, Pub/Sub) |
| `GCP_REGION` | `europe-central2` | Region |

### Opcjonalne — Aplikacja

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `APP_ENV` | `development` | `development` lub `production` |
| `LOG_LEVEL` | `INFO` | Poziom logowania |
| `PORT` | `8080` | Port serwera |

---

## 10. Deployment

### 10.1 Railway (zalecane)

**Wymagania:**
- Zmienna środowiskowa `GOOGLE_API_KEY`
- Dockerfile w repozytorium (Railway wykrywa automatycznie)

**Obraz Docker:**
- Python 3.12-slim, multi-stage build
- Model embeddingów pobierany podczas budowania obrazu (brak pobierania przy starcie kontenera)
- Cache modelu: `/app/.cache/huggingface` (`HF_HOME`)
- Użytkownik non-root: `appuser` (UID 1000)
- Port: 8080

**Serwowanie:**
- FastAPI serwuje frontend (`web/`) jako pliki statyczne pod `/`
- Wszystkie trasy API mają pierwszeństwo przed plikami statycznymi
- Jeden deployment obsługuje frontend i backend

### 10.2 GCP Cloud Run (alternatywnie)

- `cloudbuild.yaml` zawiera pipeline CI/CD
- Artifact Registry: `europe-central2-docker.pkg.dev`
- Sekrety z GCP Secret Manager (automatycznie wstrzykiwane)
- Autoscaling: 0–10 instancji, 1 CPU, 2 GiB RAM, timeout 300s

### 10.3 GitHub Pages (frontend-only)

- GitHub Actions workflow: `.github/workflows/deploy-chat.yml`
- Wyzwalacz: push do `main`, zmiany w `web/**`
- Wdraża katalog `web/` na GitHub Pages
- Frontend łączy się z osobnym backendem przez ustawienie `backendUrl`

---

## 11. Bezpieczeństwo

- Webhook Messengera: walidacja HMAC-SHA256 każdego żądania
- Docker: kontener uruchamiany jako non-root (`appuser`)
- Tokeny Allegro: przechowywane lokalnie (dev) lub w Secret Manager (prod)
- CORS: `allow_origins=["*"]` — wymaga zawężenia na produkcji
- Allegro credentials: wbudowane w kod (należy przenieść do zmiennych środowiskowych)

---

## 12. Ograniczenia i znane zachowania

- **RAG jest pusty po starcie** — baza wiedzy wymaga ręcznego zaindeksowania przez `/admin/rag/*`
- **ChromaDB jest efemeryczny na Railway** — dane tracone przy restarcie kontenera (brak persistent volume)
- **Jedno konto Allegro** — system zakłada jednego sprzedawcę; tokeny przechowywane w jednym pliku
- **Historia rozmów w dwóch miejscach** — Firestore (backend) i LocalStorage (frontend) są niezależne
- **Pub/Sub** — zintegrowany w kodzie, ale nieaktywnie używany w głównym przepływie
- **Secret Manager dla tokenów Allegro** — opcja `allegro_token_store = "secret_manager"` jest w konfiguracji, ale implementacja nie jest ukończona
