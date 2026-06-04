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
    return `Jesteś trenerem kalisteniki wojskowej. Wygeneruj plan treningu w formacie JSON.\n\nParametry:\n- Poziom: ${level}\n- Czas: ${duration} minut\n- Skupienie: ${focus.join(', ')}\n- Uwagi: ${notes || 'brak'}\n\nOdpowiedz TYLKO i wyłącznie poprawnym JSON (bez markdown, bez komentarzy):\n{\n  "name": "Nazwa treningu",\n  "duration_min": ${duration},\n  "warmup": [\n    { "name": "Nazwa ćwiczenia", "duration_sec": 30, "emoji": "🔥" }\n  ],\n  "exercises": [\n    {\n      "name": "Nazwa ćwiczenia po polsku",\n      "emoji": "💪",\n      "sets": 3,\n      "reps": 10,\n      "rest_sec": 60,\n      "tip": "Krótka wskazówka techniczna"\n    }\n  ],\n  "cooldown": [\n    { "name": "Rozciąganie", "duration_sec": 30, "emoji": "🧘" }\n  ],\n  "motivation": "Krótkie wojskowe motto na dziś"\n}\n\nWażne: emoji powinny być dobrane do ćwiczenia.`;
  }

  return { generate, buildWorkoutPrompt };
})();