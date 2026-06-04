const App = (() => {
  let _settings = { apiKey: '', model: 'gemini-2.0-flash', name: 'Żołnierz', notif: false };
  let _currentPage = 'dashboard';
  let _toastTimeout = null;

  async function boot() {
    const saved = await DB.get('settings');
    if (saved) Object.assign(_settings, saved);
    renderSettings();
    await renderDashboard();
    setTimeout(() => {
      document.getElementById('splash').style.opacity = '0';
      setTimeout(() => {
        document.getElementById('splash').classList.add('hidden');
        document.getElementById('app').classList.remove('hidden');
      }, 400);
    }, 800);
    if ('serviceWorker' in navigator)
      navigator.serviceWorker.register('sw.js').catch(() => {});
    document.querySelectorAll('.chip').forEach(btn =>
      btn.addEventListener('click', () => btn.classList.toggle('active'))
    );
  }

  function goTo(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.page === page)
    );
    document.getElementById(`page-${page}`)?.classList.add('active');
    const titles = { dashboard: 'Dashboard', generate: 'Generuj trening', workout: 'Trening', history: 'Postępy', settings: 'Ustawienia' };
    document.getElementById('page-title').textContent = titles[page] ?? page;
    document.getElementById('btn-back').classList.toggle('hidden', !['workout','settings'].includes(page));
    _currentPage = page;
    if (page === 'dashboard') renderDashboard();
    if (page === 'history')   renderHistory();
    if (page === 'settings')  renderSettings();
  }

  async function renderDashboard() {
    const workouts = await DB.getAllWorkouts();
    const { current, best } = Workout.computeStreak(workouts);
    document.getElementById('streak-count').textContent = current;
    document.getElementById('stat-total-workouts').textContent = workouts.length;
    document.getElementById('stat-best-streak').textContent = best;
    document.getElementById('stat-total-reps').textContent = workouts.reduce((s, w) => s + (w.total_reps ?? 0), 0);
    const now = new Date();
    document.getElementById('hero-date').textContent = now.toLocaleDateString('pl-PL', { weekday: 'long', day: 'numeric', month: 'long' });
    document.getElementById('hero-greeting').textContent = greeting(_settings.name);
    const { days, reps } = Workout.weeklyReps(workouts);
    Charts.weeklyBar('chart-week', days, reps);
    renderTodayCard(workouts);
  }

  function renderTodayCard(workouts) {
    const today = new Date().toISOString().split('T')[0];
    const todayDone = workouts.filter(w => w.date === today);
    const plan = Workout.getPlan();
    const box = document.getElementById('today-content');
    if (todayDone.length) {
      const w = todayDone[todayDone.length - 1];
      box.innerHTML = `<div style="color:var(--success);font-weight:700;font-size:1.1rem">✓ Trening wykonany!</div><p>${w.planName ?? 'Trening'} — ${w.total_reps ?? 0} powtórzeń</p><button class="btn btn-outline mt-1" onclick="App.goTo('generate')">Zrób kolejny</button>`;
    } else if (plan) {
      box.innerHTML = `<p><strong>${plan.name}</strong></p><p class="muted">${plan.exercises?.length ?? 0} ćwiczeń · ${plan.duration_min} min</p><button class="btn btn-primary mt-1" onclick="App.startWorkout()">▶ Zacznij</button>`;
    } else {
      box.innerHTML = `<p class="muted">Brak zaplanowanego treningu</p><button class="btn btn-primary mt-1" onclick="App.goTo('generate')">Wygeneruj z AI ✨</button>`;
    }
  }

  async function generateWorkout() {
    if (!_settings.apiKey) { toast('Ustaw klucz Gemini API w Ustawieniach ⚙'); goTo('settings'); return; }
    const level = document.getElementById('gen-level').value;
    const duration = document.getElementById('gen-duration').value;
    const focus = [...document.querySelectorAll('#gen-focus .chip.active')].map(c => c.dataset.val);
    const notes = document.getElementById('gen-notes').value.trim();
    if (!focus.length) { toast('Wybierz przynajmniej jedno skupienie'); return; }
    const resultBox = document.getElementById('gen-result');
    const loadingBox = document.getElementById('gen-loading');
    const planBox = document.getElementById('gen-plan');
    const btn = document.getElementById('btn-generate');
    resultBox.classList.remove('hidden');
    loadingBox.classList.remove('hidden');
    planBox.classList.add('hidden');
    btn.disabled = true;
    try {
      const prompt = Gemini.buildWorkoutPrompt({ level, duration, focus, notes });
      const raw = await Gemini.generate(prompt, { apiKey: _settings.apiKey, model: _settings.model });
      const plan = JSON.parse(raw.replace(/```json\n?/g,'').replace(/```\n?/g,'').trim());
      Workout.loadPlan(plan);
      renderGeneratedPlan(plan, planBox);
      loadingBox.classList.add('hidden');
      planBox.classList.remove('hidden');
      toast('Plan treningu gotowy! 💪');
    } catch (err) {
      loadingBox.classList.add('hidden');
      planBox.classList.remove('hidden');
      planBox.innerHTML = `<div class="card" style="border-color:var(--danger)"><p style="color:var(--danger)">Błąd: ${err.message}</p></div>`;
      toast(`Błąd: ${err.message}`, 5000);
    } finally { btn.disabled = false; }
  }

  function renderGeneratedPlan(plan, container) {
    const exItems = plan.exercises.map(ex =>
      `<div class="exercise-item"><div><div class="ex-name">${ex.emoji ?? ''} ${ex.name}</div><div class="ex-desc">${ex.tip ?? ''}</div></div><div class="ex-sets">${ex.sets}×${ex.reps}<br><span style="color:var(--muted);font-size:.7rem">${ex.rest_sec}s odpocz.</span></div></div>`
    ).join('');
    container.innerHTML = `<div class="gen-plan-card"><h3>${plan.name}</h3><p class="muted" style="margin-bottom:.75rem">⏱ ${plan.duration_min} min · ${plan.exercises.length} ćwiczeń</p>${plan.warmup?.length ? `<p style="color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:.4rem">Rozgrzewka</p>${plan.warmup.map(w => `<div class="exercise-item"><div>${w.emoji} ${w.name}</div><div class="ex-sets">${w.duration_sec}s</div></div>`).join('')}<hr style="border-color:var(--border);margin:.75rem 0">` : ''}<p style="color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:.4rem">Ćwiczenia</p>${exItems}${plan.cooldown?.length ? `<hr style="border-color:var(--border);margin:.75rem 0"><p style="color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:.4rem">Schładzanie</p>${plan.cooldown.map(c => `<div class="exercise-item"><div>${c.emoji} ${c.name}</div><div class="ex-sets">${c.duration_sec}s</div></div>`).join('')}` : ''}${plan.motivation ? `<div style="margin-top:.75rem;padding:.75rem;background:var(--bg3);border-radius:8px;font-style:italic;color:var(--accent);text-align:center">"${plan.motivation}"</div>` : ''}<button class="btn btn-primary w-full" style="margin-top:.75rem" onclick="App.startWorkout()">▶ Rozpocznij trening</button></div>`;
  }

  function startWorkout() {
    const plan = Workout.getPlan();
    if (!plan) { toast('Brak planu treningu'); return; }
    const session = Workout.startSession(plan);
    document.getElementById('workout-name').textContent = plan.name;
    const list = document.getElementById('exercise-list');
    list.innerHTML = session.exercises.map((ex, ei) =>
      `<div class="ex-tracker"><div class="ex-tracker-header"><div><div class="ex-visual">${ex.emoji}</div><div class="ex-tracker-name">${ex.name}</div><div class="ex-tracker-target">${ex.sets} serie × ${ex.reps} powt. · ${ex.rest_sec}s przerwy</div></div></div>${ex.tip ? `<p class="muted" style="font-size:.8rem;margin-bottom:.5rem">${ex.tip}</p>` : ''}<div class="sets-row"><span class="set-label">Serie:</span>${Array.from({length:ex.sets},(_,si) => `<button class="set-btn" id="set-${ei}-${si}" onclick="App.toggleSet(${ei},${si})">${si+1}</button>`).join('')}</div></div>`
    ).join('');
    goTo('workout');
  }

  function toggleSet(ei, si) {
    Workout.toggleSet(ei, si);
    document.getElementById(`set-${ei}-${si}`)?.classList.toggle('done');
  }

  async function finishWorkout() {
    if (!confirm('Zakończyć trening?')) return;
    const record = await Workout.finishSession();
    if (!record) return;
    toast(`Trening ukończony! ${record.total_reps} powtórzeń 🎖`);
    goTo('dashboard');
  }

  async function renderHistory() {
    const workouts = await DB.getAllWorkouts();
    const list = document.getElementById('history-list');
    if (!workouts.length) { list.innerHTML = '<div class="card"><p class="muted">Brak zapisanych treningów.</p></div>'; return; }
    const allNames = [...new Set(workouts.flatMap(w => (w.exercises ?? []).map(e => e.name)))].sort();
    const sel = document.getElementById('hist-exercise-select');
    sel.innerHTML = '<option value="">— wybierz ćwiczenie —</option>' + allNames.map(n => `<option value="${n}">${n}</option>`).join('');
    list.innerHTML = [...workouts].reverse().slice(0, 30).map(w =>
      `<div class="history-item"><div><div class="hist-date">${formatDate(w.date)}</div><div class="hist-name">${w.planName ?? 'Trening'}</div></div><div><div class="hist-badge">${w.total_reps ?? 0} powt.</div><div class="hist-date" style="text-align:right;margin-top:.2rem">${formatDuration(w.duration_sec)}</div></div></div>`
    ).join('');
  }

  async function updateProgressChart() {
    const name = document.getElementById('hist-exercise-select').value;
    if (!name) return;
    const workouts = await DB.getAllWorkouts();
    const prog = Workout.exerciseProgress(workouts, name);
    Charts.progressLine('chart-progress', prog.map(p => formatDate(p.date)), prog.map(p => p.reps), name);
  }

  function renderSettings() {
    document.getElementById('set-api-key').value = _settings.apiKey ?? '';
    document.getElementById('set-model').value   = _settings.model  ?? 'gemini-2.0-flash';
    document.getElementById('set-name').value    = _settings.name   ?? '';
    document.getElementById('set-notif').checked = _settings.notif  ?? false;
  }

  async function saveSettings() {
    _settings.apiKey = document.getElementById('set-api-key').value.trim();
    _settings.model  = document.getElementById('set-model').value;
    _settings.name   = document.getElementById('set-name').value.trim() || 'Żołnierz';
    _settings.notif  = document.getElementById('set-notif').checked;
    await DB.set('settings', _settings);
    if (_settings.notif && 'Notification' in window && Notification.permission === 'default')
      Notification.requestPermission().then(p => { if (p === 'granted') toast('Powiadomienia włączone 🔔'); });
    toast('Ustawienia zapisane ✓');
  }

  async function clearData() {
    if (!confirm('Usunąć WSZYSTKIE dane treningowe? Tej operacji nie można cofnąć.')) return;
    await DB.clearAll();
    Workout.loadPlan(null);
    toast('Dane usunięte');
    goTo('dashboard');
  }

  function toast(msg, duration = 3000) {
    clearTimeout(_toastTimeout);
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.classList.remove('hidden');
    _toastTimeout = setTimeout(() => el.classList.add('hidden'), duration);
  }

  function greeting(name) {
    const h = new Date().getHours();
    if (h < 6)  return `Dobranoc, ${name}.`;
    if (h < 12) return `Dzień dobry, ${name}! Czas na ćwiczenia 💪`;
    if (h < 18) return `Dobry dzień, ${name}! Nie odkładaj treningu!`;
    return `Dobry wieczór, ${name}! Wieczorny trening?`;
  }

  function formatDate(iso) {
    if (!iso) return '';
    return new Date(iso + 'T00:00:00').toLocaleDateString('pl-PL', { day: 'numeric', month: 'short', year: 'numeric' });
  }

  function formatDuration(sec) {
    if (!sec) return '';
    return `${Math.floor(sec/60)}min ${sec%60}s`;
  }

  document.getElementById('btn-back').addEventListener('click', () => {
    if (_currentPage === 'workout') { if (confirm('Przerwać trening?')) { Workout.finishSession(); goTo('dashboard'); } }
    else goTo('dashboard');
  });
  document.getElementById('btn-settings').addEventListener('click', () => goTo('settings'));

  window.addEventListener('DOMContentLoaded', boot);

  return { goTo, generateWorkout, startWorkout, toggleSet, finishWorkout, saveSettings, clearData, updateProgressChart };
})();