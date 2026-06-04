/* Chart.js helpers */
const Charts = (() => {
  const instances = {};

  const DEFAULTS = {
    color:     '#4caf50',
    colorFade: 'rgba(76,175,80,0.15)',
    grid:      'rgba(42,74,42,0.5)',
    text:      '#6a8f6a',
    font:      "'Segoe UI', system-ui, sans-serif"
  };

  function destroy(id) {
    if (instances[id]) { instances[id].destroy(); delete instances[id]; }
  }

  function weeklyBar(canvasId, labels, data) {
    destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor: data.map((v, i) =>
            i === data.length - 1 ? DEFAULTS.color : 'rgba(76,175,80,0.35)'
          ),
          borderRadius: 6,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: DEFAULTS.text, font: { family: DEFAULTS.font, size: 11 } },
               grid:  { color: DEFAULTS.grid } },
          y: { ticks: { color: DEFAULTS.text, font: { family: DEFAULTS.font, size: 11 } },
               grid:  { color: DEFAULTS.grid }, beginAtZero: true }
        }
      }
    });
  }

  function progressLine(canvasId, labels, data, exerciseName) {
    destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;

    instances[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: exerciseName,
          data,
          borderColor: DEFAULTS.color,
          backgroundColor: DEFAULTS.colorFade,
          fill: true,
          tension: 0.4,
          pointBackgroundColor: DEFAULTS.color,
          pointRadius: 5
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: DEFAULTS.text, font: { family: DEFAULTS.font } } }
        },
        scales: {
          x: { ticks: { color: DEFAULTS.text, font: { family: DEFAULTS.font, size: 11 } },
               grid:  { color: DEFAULTS.grid } },
          y: { ticks: { color: DEFAULTS.text, font: { family: DEFAULTS.font, size: 11 } },
               grid:  { color: DEFAULTS.grid }, beginAtZero: true }
        }
      }
    });
  }

  return { weeklyBar, progressLine, destroy };
})();
