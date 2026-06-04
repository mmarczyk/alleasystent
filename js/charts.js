const Charts = (() => {
  const instances = {};
  const D = {
    color: '#4caf50', colorFade: 'rgba(76,175,80,0.15)',
    grid: 'rgba(42,74,42,0.5)', text: '#6a8f6a',
    font: "'Segoe UI', system-ui, sans-serif"
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
      data: { labels, datasets: [{ data,
        backgroundColor: data.map((v, i) => i === data.length - 1 ? D.color : 'rgba(76,175,80,0.35)'),
        borderRadius: 6, borderSkipped: false }] },
      options: { responsive: true, plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: D.text, font: { family: D.font, size: 11 } }, grid: { color: D.grid } },
          y: { ticks: { color: D.text, font: { family: D.font, size: 11 } }, grid: { color: D.grid }, beginAtZero: true }
        }}
    });
  }

  function progressLine(canvasId, labels, data, name) {
    destroy(canvasId);
    const ctx = document.getElementById(canvasId)?.getContext('2d');
    if (!ctx) return;
    instances[canvasId] = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets: [{ label: name, data,
        borderColor: D.color, backgroundColor: D.colorFade,
        fill: true, tension: 0.4, pointBackgroundColor: D.color, pointRadius: 5 }] },
      options: { responsive: true,
        plugins: { legend: { labels: { color: D.text, font: { family: D.font } } } },
        scales: {
          x: { ticks: { color: D.text, font: { family: D.font, size: 11 } }, grid: { color: D.grid } },
          y: { ticks: { color: D.text, font: { family: D.font, size: 11 } }, grid: { color: D.grid }, beginAtZero: true }
        }}
    });
  }

  return { weeklyBar, progressLine, destroy };
})();