# Zarządzanie funkcjonalnościami i zgłoszeniami

## Jak zgłosić nową funkcję

Używamy **GitHub Issues** jako jedynego miejsca do zgłaszania pomysłów, błędów i zadań.

### 1. Sprawdź czy issue już istnieje

Zanim otworzysz nowe zgłoszenie, przeszukaj istniejące Issues — może ktoś już to zgłosił.

### 2. Otwórz nowe Issue

Wybierz odpowiedni szablon:

| Typ | Szablon | Kiedy używać |
|---|---|---|
| Nowa funkcja | `feature_request.md` | Pomysł na nową funkcjonalność lub ulepszenie |
| Błąd | `bug_report.md` | Coś działa niepoprawnie |

### 3. Etykiety (Labels)

Każde Issue otrzymuje etykiety automatycznie lub ręcznie:

| Etykieta | Opis |
|---|---|
| `feature-request` | Nowa funkcjonalność |
| `bug` | Błąd w działaniu |
| `needs-triage` | Nowe zgłoszenie — wymaga oceny |
| `accepted` | Zatwierdzone do implementacji |
| `in-progress` | W trakcie pracy |
| `blocked` | Zablokowane (dependencje, zewnętrzne API) |
| `wont-fix` | Nie będzie implementowane (z uzasadnieniem) |
| `allegro` | Dotyczy integracji Allegro |
| `rag` | Dotyczy bazy wiedzy |
| `pwa` | Dotyczy interfejsu webowego |
| `messenger` | Dotyczy kanału Facebook Messenger |

---

## Proces oceny zgłoszeń (triage)

```
Nowe Issue
    │
    ▼
[needs-triage] → Ocena w ciągu 1 tygodnia
    │
    ├── Duplikat → zamknij, wskaż oryginał
    ├── Niejasne → poproś o doprecyzowanie
    ├── Odrzucone → [wont-fix] + uzasadnienie
    └── Zatwierdzone → [accepted] + przypisanie do milestone
                              │
                              ▼
                        [in-progress]
                              │
                              ▼
                        Pull Request
                              │
                              ▼
                          Merge → zamknięcie Issue
```

---

## Priorytety i milestone'y

Planowanie pracy odbywa się przez **GitHub Milestones**. Każdy milestone odpowiada wersji lub sprintowi.

Kryteria priorytetu:

1. **Krytyczna** — błąd blokujący działanie systemu lub bezpieczeństwo
2. **Wysoka** — funkcja wymagana przez aktywnych użytkowników
3. **Średnia** — ulepszenie z jasną wartością
4. **Niska** — "nice to have", realizowane gdy jest czas

---

## Propozycje większych zmian (RFC)

Dla znaczących zmian architektonicznych lub nowych obszarów funkcjonalnych (np. nowy kanał komunikacji, zmiana modelu AI, integracja z nową platformą) — otwórz Issue z etykietą `rfc` i opisz:

- Cel i motywację
- Proponowane podejście
- Wpływ na istniejący kod
- Alternatywy, które rozważałeś

RFC wymaga dyskusji i akceptacji przed rozpoczęciem implementacji.

---

## Backlog a roadmapa

- **Backlog** — wszystkie zatwierdzone (`accepted`) Issues bez przypisanego milestone
- **Roadmapa** — Issues przypisane do konkretnych milestone'ów

Backlog jest regularnie przeglądany i priorytetyzowany.
