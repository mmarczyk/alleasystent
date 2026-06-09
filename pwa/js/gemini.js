/* Gemini API client */
const Gemini = (() => {
  const BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

  async function generate(prompt, { apiKey, model = 'gemini-2.0-flash' } = {}) {
    if (!apiKey) throw new Error('Brak klucza API Gemini. Ustaw go w Ustawieniach ⚙');

    const url = `${BASE}/${model}:generateContent?key=${apiKey}`;
    const body = {
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.7, maxOutputTokens: 2048 }
    };

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message ?? `HTTP ${res.status}`);
    }

    const data = await res.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
  }

  function buildWorkoutPrompt({ level, duration, focus, notes }) {
    return `Jesteś trenerem kalisteniki wojskowej. Wygeneruj plan treningu w formacie JSON.

Parametry:
- Poziom: ${level}
- Czas: ${duration} minut
- Skupienie: ${focus.join(', ')}
- Uwagi: ${notes || 'brak'}

Odpowiedz TYLKO i wyłącznie poprawnym JSON (bez markdown, bez komentarzy):
{
  "name": "Nazwa treningu",
  "duration_min": ${duration},
  "warmup": [
    { "name": "Nazwa ćwiczenia", "duration_sec": 30, "emoji": "🔥" }
  ],
  "exercises": [
    {
      "name": "Nazwa ćwiczenia po polsku",
      "emoji": "💪",
      "sets": 3,
      "reps": 10,
      "rest_sec": 60,
      "tip": "Krótka wskazówka techniczna"
    }
  ],
  "cooldown": [
    { "name": "Rozciąganie", "duration_sec": 30, "emoji": "🧘" }
  ],
  "motivation": "Krótkie wojskowe motto na dziś"
}

Ważne: emoji powinny być dobrane do ćwiczenia. Pompki=💪, podciąganie=🏋️, przysiady=🦵, brzuszki=🔥, plank=⚔️, burpee=💥, dipy=🤜, wspięcia=🦶, deska boczna=🛡️`;
  }

  return { generate, buildWorkoutPrompt };
})();
