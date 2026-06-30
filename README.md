# AllEasystent

Asystent AI dla sprzedawców na Allegro. Umożliwia zarządzanie zamówieniami, ofertami, wiadomościami i kontem przez interfejs konwersacyjny — bez klikania w panelu Allegro.

Obsługuje wielokanałowy dostęp: **interfejs webowy (PWA)** oraz **Facebook Messenger**.

---

## Funkcje

- Zarządzanie zamówieniami, ofertami, wiadomościami i kontem Allegro przez chat
- Inteligentny routing zapytań — reguły słów kluczowych + Gemini jako fallback
- Baza wiedzy sklepu (RAG) — odpowiedzi na pytania o polityki, FAQ, produkty
- PWA z historią rozmów, trybem offline i eksportem do pliku
- Integracja z Facebook Messengerem przez webhook
- Powiadomienia push (VAPID — iOS 16.4+, Android, desktop)
- Obsługa języka polskiego i angielskiego

---

## Szybki start

### Wymagania

- Python 3.12+
- Klucz Google AI Studio (`GOOGLE_API_KEY`)

### Lokalnie

```bash
git clone https://github.com/mmarczyk/alleasystent
cd alleasystent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Uzupełnij GOOGLE_API_KEY w pliku .env

uvicorn main:app --reload --port 8080
```

Aplikacja dostępna pod `http://localhost:8080`.

### Docker

```bash
docker build -t alleasystent .
docker run -p 8080:8080 -e GOOGLE_API_KEY=... alleasystent
```

---

## Konfiguracja

Wszystkie parametry przez plik `.env` lub zmienne środowiskowe. Szczegółowa lista w [`REQUIREMENTS.md`](./REQUIREMENTS.md#konfiguracja).

| Zmienna | Wymagana | Opis |
|---|---|---|
| `GOOGLE_API_KEY` | **tak** | Klucz Google AI Studio (Gemini) |
| `ALLEGRO_CLIENT_ID` | nie | Client ID aplikacji Allegro |
| `ALLEGRO_CLIENT_SECRET` | nie | Client Secret Allegro |
| `FACEBOOK_PAGE_ACCESS_TOKEN` | nie | Token strony Facebook (Messenger) |
| `GCP_PROJECT_ID` | nie | Projekt GCP (Firestore, Secret Manager) |

---

## Architektura

```
Kanały wejścia: PWA (web/)  │  Facebook Messenger
                             │
                    FastAPI (main.py)
                             │
                       Orchestrator
                    ┌────────┴────────┐
               AllegroAgent      RAGAgent
               (Gemini Flash)  (Gemini Flash)
                    │
            AllegroService → Allegro REST API
```

Szczegółowy opis architektury i wymagań funkcjonalnych: [`REQUIREMENTS.md`](./REQUIREMENTS.md).

---

## Deployment

### Railway (zalecane)

1. Stwórz nowy projekt w Railway i połącz z tym repozytorium
2. Dodaj zmienną środowiskową `GOOGLE_API_KEY`
3. Railway wykryje `Dockerfile` automatycznie i wdroży aplikację

### GCP Cloud Run

```bash
gcloud builds submit --config cloudbuild.yaml
```

CI/CD skonfigurowany w `cloudbuild.yaml`. Sekrety pobierane z GCP Secret Manager.

### GitHub Pages (frontend-only)

Push do `main` automatycznie wdraża katalog `web/` na GitHub Pages (`.github/workflows/deploy-chat.yml`). Frontend wymaga ustawienia `backendUrl` w ustawieniach aplikacji.

---

## API

| Metoda | Ścieżka | Opis |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Zapytanie do asystenta |
| `GET` | `/allegro/auth` | Inicjuje autoryzację Allegro (OAuth2 Device Flow) |
| `GET` | `/allegro/auth/status` | Stan autoryzacji |
| `POST` | `/webhook/facebook` | Zdarzenia Messengera |
| `POST` | `/admin/rag/index-file` | Indeksowanie pliku do bazy wiedzy |
| `POST` | `/admin/rag/index-faq` | Indeksowanie FAQ |
| `POST` | `/admin/rag/index-allegro-offers` | Indeksowanie ofert z Allegro |

---

## Struktura projektu

```
alleasystent/
├── agents/                  # Agenci AI
│   ├── orchestrator.py      # Router + klasyfikacja intencji
│   ├── base_agent.py        # Klasa bazowa agentów
│   ├── allegro/             # Agent Allegro (zamówienia, oferty, wiadomości)
│   ├── communication/       # Facebook Messenger
│   └── rag/                 # RAG agent + retriever + indexer
├── config/settings.py       # Konfiguracja (Pydantic)
├── models/                  # Modele danych
├── services/                # Klienty zewnętrznych API
├── webhooks/                # Handlery webhooku Facebook
├── web/                     # Frontend PWA
├── deployment/              # Skrypty GCP
├── main.py                  # Aplikacja FastAPI
├── Dockerfile
├── requirements.txt
├── PRD.md                   # Product Requirements Document
└── REQUIREMENTS.md          # Wymagania funkcjonalne (ustrukturyzowane)
```

---

## Testy

Testy zostały przeniesione do osobnego repozytorium [`alleasystent-qa`](https://github.com/mmarczyk/alleasystent-qa), które zawiera pakiet pytest oraz aplikację webową do uruchamiania testów.

---

## Znane ograniczenia

- RAG jest pusty po starcie — wymaga ręcznego zaindeksowania przez `/admin/rag/*`
- ChromaDB traci dane przy restarcie kontenera (brak persistent volume na Railway)
- System obsługuje jedno konto Allegro na instancję
- CORS: `allow_origins=["*"]` — zawęzić na produkcji
- `allegro_token_store = "secret_manager"` jest w konfiguracji, ale implementacja nie jest ukończona
