/* Workout session manager */
const Workout = (() => {
  let _plan     = null;   // current generated plan
  let _session  = null;   // active workout session
  let _timer    = null;
  let _elapsed  = 0;

  function loadPlan(plan) {
    _plan = plan;
  }

  function getPlan() { return _plan; }

  function startSession(plan) {
    _plan    = plan;
    _elapsed = 0;
    _session = {
      planName:  plan.name,
      startedAt: new Date().toISOString(),
      date:      new Date().toISOString().split('T')[0],
      exercises: plan.exercises.map(ex => ({
        name:     ex.name,
        emoji:    ex.emoji,
        sets:     ex.sets,
        reps:     ex.reps,
        rest_sec: ex.rest_sec,
        tip:      ex.tip,
        done:     Array(ex.sets).fill(false)  // which sets completed
      }))
    };

    // Start timer
    clearInterval(_timer);
    _timer = setInterval(() => {
      _elapsed++;
      const m = String(Math.floor(_elapsed / 60)).padStart(2, '0');
      const s = String(_elapsed % 60).padStart(2, '0');
      const el = document.getElementById('workout-timer');
      if (el) el.textContent = `${m}:${s}`;
    }, 1000);

    return _session;
  }

  function toggleSet(exerciseIndex, setIndex) {
    if (!_session) return;
    _session.exercises[exerciseIndex].done[setIndex] =
      !_session.exercises[exerciseIndex].done[setIndex];
  }

  async function finishSession() {
    if (!_session) return null;
    clearInterval(_timer);

    const totalReps = _session.exercises.reduce((sum, ex) => {
      const doneSets = ex.done.filter(Boolean).length;
      return sum + doneSets * ex.reps;
    }, 0);

    const completedExercises = _session.exercises.filter(ex =>
      ex.done.some(Boolean)
    );

    const record = {
      ..._session,
      duration_sec:  _elapsed,
      total_reps:    totalReps,
      completed_exercises: completedExercises.length,
      total_exercises:     _session.exercises.length
    };

    await DB.saveWorkout(record);
    _session = null;
    _elapsed = 0;
    return record;
  }

  function getSession() { return _session; }

  /* Compute streak from sorted workout dates */
  function computeStreak(workouts) {
    if (!workouts.length) return { current: 0, best: 0 };

    const dates = [...new Set(workouts.map(w => w.date))].sort().reverse();
    const today = new Date().toISOString().split('T')[0];

    let current = 0;
    let best    = 0;
    let cursor  = today;

    for (const d of dates) {
      if (d === cursor) {
        current++;
        if (current > best) best = current;
        // move cursor back one day
        const dt = new Date(cursor);
        dt.setDate(dt.getDate() - 1);
        cursor = dt.toISOString().split('T')[0];
      } else {
        break;
      }
    }

    // best over full history
    let streak = 1; let maxStreak = 1;
    for (let i = 1; i < dates.length; i++) {
      const prev = new Date(dates[i - 1]);
      const curr = new Date(dates[i]);
      const diff = (prev - curr) / 86400000;
      if (diff === 1) { streak++; if (streak > maxStreak) maxStreak = streak; }
      else streak = 1;
    }
    best = Math.max(best, maxStreak);

    return { current, best };
  }

  /* Last 7 days rep counts for chart */
  function weeklyReps(workouts) {
    const days  = [];
    const reps  = [];
    for (let i = 6; i >= 0; i--) {
      const dt = new Date();
      dt.setDate(dt.getDate() - i);
      const key = dt.toISOString().split('T')[0];
      days.push(dt.toLocaleDateString('pl-PL', { weekday: 'short' }));
      const dayReps = workouts
        .filter(w => w.date === key)
        .reduce((s, w) => s + (w.total_reps ?? 0), 0);
      reps.push(dayReps);
    }
    return { days, reps };
  }

  /* Per-exercise progress for history chart */
  function exerciseProgress(workouts, exerciseName) {
    const relevant = workouts
      .filter(w => w.exercises?.some(e => e.name === exerciseName))
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(-20);

    return relevant.map(w => {
      const ex = w.exercises.find(e => e.name === exerciseName);
      return {
        date:  w.date,
        reps:  (ex?.done?.filter(Boolean).length ?? 0) * (ex?.reps ?? 0)
      };
    });
  }

  return {
    loadPlan, getPlan,
    startSession, toggleSet, finishSession, getSession,
    computeStreak, weeklyReps, exerciseProgress
  };
})();
