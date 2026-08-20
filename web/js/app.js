/* ═══════════════════════════════════════════════════
   AllEasystent Chat UI — main controller
   ═══════════════════════════════════════════════════ */

// ── Marked.js config ─────────────────────────────
if (typeof marked !== 'undefined') {
  marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: (code, lang) => {
      if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return typeof hljs !== 'undefined' ? hljs.highlightAuto(code).value : code;
    }
  });
}

// ── Update detector ───────────────────────────────
// Single source of truth: the service worker lifecycle (registered further
// down, near checkAuth). sw.js activates a new version as soon as it's
// installed, so this banner is purely informational — it gives the user a
// moment to notice before the auto-reload below fires.
const AppUpdater = (() => {
  let _bannerShown = false;

  function showBanner() {
    if (_bannerShown) return;
    _bannerShown = true;
    const banner = document.createElement('div');
    banner.id = 'update-banner';
    banner.style.cssText = [
      'position:fixed', 'top:0', 'left:0', 'right:0', 'z-index:9999',
      'background:#2563eb', 'color:#fff', 'text-align:center',
      'padding:.6rem 1rem', 'font-size:.9rem', 'font-weight:500',
      'display:flex', 'align-items:center', 'justify-content:center', 'gap:.75rem',
    ].join(';');
    banner.innerHTML = '🔄 Dostępna nowa wersja aplikacji. '
      + '<button onclick="AppUpdater.reload()" style="background:#fff;color:#2563eb;border:none;'
      + 'border-radius:4px;padding:.25rem .75rem;font-weight:700;cursor:pointer">Odśwież teraz</button>';
    document.body.prepend(banner);
    OfflineBanner.reposition();
    // Auto-reload after 10 s if user hasn't clicked
    setTimeout(() => AppUpdater.reload(), 10000);
  }

  function reload() {
    window.location.reload();
  }

  return { reload, showBanner };
})();

// ── Offline banner ────────────────────────────────
// navigator.onLine / the online-offline events reflect whether the device has
// a live network interface (airplane mode, no wifi/cellular) — exactly the
// "no internet at all" case this banner is for. The app shell and cached data
// (chat history, notifications) already come from localStorage/the SW cache,
// so there's nothing to fetch here — just tell the user why things look static.
const OfflineBanner = (() => {
  let el = null;

  function _reposition() {
    if (!el) return;
    const updateBanner = document.getElementById('update-banner');
    el.style.top = updateBanner ? updateBanner.offsetHeight + 'px' : '0';
  }

  function show() {
    if (el) { _reposition(); return; }
    el = document.createElement('div');
    el.id = 'offline-banner';
    el.style.cssText = [
      'position:fixed', 'left:0', 'right:0', 'z-index:9998',
      'background:#57534e', 'color:#fff', 'text-align:center',
      'padding:.5rem 1rem', 'font-size:.85rem', 'font-weight:500',
    ].join(';');
    el.textContent = '📡 Tryb offline — pokazuję zapisane dane. Połączenie zostanie wznowione automatycznie.';
    document.body.prepend(el);
    _reposition();
  }

  function hide() {
    el?.remove();
    el = null;
  }

  function _sync() {
    if (navigator.onLine === false) show(); else hide();
  }

  function init() {
    window.addEventListener('online', _sync);
    window.addEventListener('offline', _sync);
    _sync();
  }

  return { init, show, hide, reposition: _reposition };
})();

// ── Version info ──────────────────────────────────
let _backendVersion = null;

function _shortVersion(v) {
  return v && v.length > 7 ? v.slice(0, 7) : (v || 'dev');
}

function updateVersionInfo() {
  const el = document.getElementById('version-info');
  if (!el) return;
  const fe = _shortVersion(window.__FRONTEND_VERSION__);
  const be = _backendVersion ? _shortVersion(_backendVersion) : '…';
  el.textContent = `Frontend: ${fe} · Backend: ${be}`;
}

// ── Auth check ────────────────────────────────────
// ── Container wake-up ────────────────────────────
// Fire a lightweight /health ping that starts the container without blocking
// anything. Call this as early as possible so the container is warm by the
// time the user's first real API request lands.
function wakeContainer() {
  fetch(Settings.api('/health'), { credentials: 'include' })
    .then(r => r.json().catch(() => null))
    .then(data => {
      if (data?.git_sha) {
        _backendVersion = data.git_sha;
        updateVersionInfo();
      }
    })
    .catch(() => {});
}

function _applyAuthUser(user) {
  window._currentUser = user;
  document.getElementById('login-overlay').style.display = 'none';
  document.getElementById('app').style.display = '';
  const userEl = document.getElementById('user-info');
  if (userEl) {
    userEl.innerHTML = `<span style="font-size:1.1rem">🛒</span> <span style="overflow:hidden;text-overflow:ellipsis;font-weight:500">${user.name}</span>`;
  }
}

async function checkAuth() {
  // ── Fast path: valid JWT in localStorage ──────────────────────────────────
  // Decode the payload (base64, no network) — exp and name are embedded.
  // Show the app immediately, then fire /health to wake the container so
  // it is ready for the first real chat/API request.
  const token = Auth.getToken();
  if (token) {
    try {
      const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')));
      if (payload.exp > Date.now() / 1000 + 30) {
        _applyAuthUser({ sub: payload.sub, name: payload.name || payload.sub });
        wakeContainer();   // warm the container in the background — don't wait
        return true;
      }
    } catch {}
    // Token present but expired or malformed — clear it
    Auth.clearToken();
  }

  // ── No valid token ────────────────────────────────────────────────────────
  // Split deployment (GitHub Pages → Cloud Run, backendUrl set): JWT is the
  // only auth mechanism — no JWT means not logged in, show login immediately.
  //
  // Same-origin deployment (no backendUrl): session cookie might still be valid
  // (e.g. server-side /allegro/callback flow) — verify once via /auth/me.
  if (!Settings.get('backendUrl')) {
    try {
      const res = await fetch('/auth/me', { credentials: 'include' });
      if (res.ok) {
        const user = await res.json();
        // Persist the JWT so next visit is instant
        // (server doesn't re-issue JWT here, just confirm the cookie)
        _applyAuthUser(user);
        wakeContainer();
        return true;
      }
    } catch {}
  }
  document.getElementById('login-overlay').style.display = 'flex';
  return false;
}

// ── Session token (Safari ITP workaround) ────────
// Safari blocks cross-site Set-Cookie responses (ITP), so in split deployment
// (GitHub Pages → Cloud Run) we store the JWT in localStorage and send it as
// a Bearer token.  Chrome/Firefox still use the cookie automatically.
const Auth = (() => {
  const KEY = 'ae_session_token';
  function getToken() { try { return localStorage.getItem(KEY); } catch { return null; } }
  function setToken(t) { try { if (t) localStorage.setItem(KEY, t); } catch {} }
  function clearToken() { try { localStorage.removeItem(KEY); } catch {} }
  // Returns headers object with Authorization if a token is stored.
  function headers() {
    const t = getToken();
    return t ? { Authorization: 'Bearer ' + t } : {};
  }
  return { getToken, setToken, clearToken, headers };
})();

// ── Shared render helpers ─────────────────────────
// Global (not nested in Chat/DocViewer) since both modules need them.
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderMarkdown(text) {
  if (typeof marked === 'undefined') return escHtml(text).replace(/\n/g, '<br>');
  return marked.parse(text);
}

// ── Document Viewer ──────────────────────────────
// Full-screen tab-based viewer for long responses (> 500 chars).
const DocViewer = (() => {
  const _tabs = [];  // [{id, title, content, kind}]
  let _activeId = null;
  let _nextId = 0;
  const _registry = {};  // key → {content, kind}, for "Pełny widok" buttons on existing bubbles

  function _titleFromContent(content) {
    const heading = content.match(/^#{1,3}\s+(.+)/m);
    if (heading) return heading[1].replace(/[*`]/g, '').trim().slice(0, 60);
    return content.replace(/[#*`_[\]]/g, '').trim().slice(0, 60);
  }

  // kind: 'table' | 'dashboard' | 'document' | 'chat' — drives presentation in _render()
  function register(content, kind) {
    const key = ++_nextId;
    _registry[key] = { content, kind };
    return key;
  }

  function openFromKey(key) {
    const entry = _registry[key];
    if (!entry) return;
    // Auto-open (new replies) and the "Pełny widok" button on that same bubble
    // both resolve to this key — reuse the existing tab instead of stacking
    // duplicates when the user reopens something already open.
    const existing = _tabs.find(t => t.regKey === key);
    if (existing) {
      _activeId = existing.id;
      _render();
      document.getElementById('doc-viewer').classList.remove('hidden');
      return;
    }
    open(_titleFromContent(entry.content), entry.content, entry.kind, key);
  }

  function open(title, content, kind, regKey) {
    const id = ++_nextId;
    _tabs.push({ id, title: (title || _titleFromContent(content)).slice(0, 60), content, kind, regKey });
    _activeId = id;
    _render();
    document.getElementById('doc-viewer').classList.remove('hidden');
  }

  function setActive(id) {
    _activeId = id;
    _render();
  }

  function closeTab(id) {
    const idx = _tabs.findIndex(t => t.id === id);
    if (idx < 0) return;
    _tabs.splice(idx, 1);
    if (!_tabs.length) { close(); return; }
    if (_activeId === id) _activeId = _tabs[Math.min(idx, _tabs.length - 1)].id;
    _render();
  }

  function close() {
    document.getElementById('doc-viewer').classList.add('hidden');
  }

  async function copyActive() {
    const active = _tabs.find(t => t.id === _activeId);
    if (!active) return;
    try {
      await navigator.clipboard.writeText(active.content);
      UI.toast('Skopiowano!', 2000);
    } catch { UI.toast('Nie można skopiować', 2000); }
  }

  function _render() {
    const tabList = document.getElementById('doc-tab-list');
    if (!tabList) return;
    tabList.innerHTML = _tabs.map(t =>
      `<button class="doc-tab${t.id === _activeId ? ' active' : ''}" onclick="DocViewer.setActive(${t.id})">` +
        `<span class="doc-tab-name">📄 ${_esc(t.title)}</span>` +
        `<button class="doc-tab-x" onclick="event.stopPropagation();DocViewer.closeTab(${t.id})">✕</button>` +
      `</button>`
    ).join('');

    const active = _tabs.find(t => t.id === _activeId);
    const content = document.getElementById('doc-content');
    if (!content) return;
    content.innerHTML = active ? renderMarkdown(active.content) : '';
    content.dataset.kind = active?.kind || '';
    if (active?.kind === 'dashboard') _wrapDashboardSections(content);
    _renderCharts(content);
    if (typeof hljs !== 'undefined') {
      content.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
    }
  }

  // ── Charts ────────────────────────────────────────
  // Dashboard-format replies may embed one or more ```chart fenced JSON blocks
  // (see agents/orchestrator.py _FORMAT_PREFIXES["dashboard"]). Marked renders
  // those as <pre><code class="language-chart">...</code></pre> — swap each
  // one for a live Chart.js canvas instead of showing raw JSON.
  const CHART_PALETTE = ['#818cf8', '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#84cc16'];

  function _chartConfig(spec) {
    const type = ['bar', 'line', 'pie', 'doughnut'].includes(spec.type) ? spec.type : 'bar';
    const isSliced = type === 'pie' || type === 'doughnut';
    const labels = Array.isArray(spec.labels) ? spec.labels : [];
    const series = Array.isArray(spec.series) ? spec.series : [];
    const datasets = series.map((s, i) => ({
      label: s.name || `Seria ${i + 1}`,
      data: Array.isArray(s.data) ? s.data : [],
      backgroundColor: isSliced
        ? labels.map((_, j) => CHART_PALETTE[j % CHART_PALETTE.length])
        : CHART_PALETTE[i % CHART_PALETTE.length],
      borderColor: isSliced ? '#0f0f1a' : CHART_PALETTE[i % CHART_PALETTE.length],
      borderWidth: isSliced ? 2 : (type === 'line' ? 2 : 1),
      fill: type === 'line' ? false : true,
      tension: .3,
    }));
    return {
      type,
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: datasets.length > 1 || isSliced,
            labels: { color: '#e2e8f0' },
          },
        },
        scales: isSliced ? {} : {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,.12)' } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,.12)' }, beginAtZero: true },
        },
      },
    };
  }

  function _renderCharts(container) {
    if (typeof Chart === 'undefined') return;
    container.querySelectorAll('code[class*="language-chart"]').forEach(codeEl => {
      let spec;
      try { spec = JSON.parse(codeEl.textContent); } catch { return; }
      const pre = codeEl.closest('pre');
      if (!pre) return;

      const wrap = document.createElement('div');
      wrap.className = 'chart-wrap';
      if (spec.title) {
        const title = document.createElement('div');
        title.className = 'chart-title';
        title.textContent = spec.title;
        wrap.appendChild(title);
      }
      const canvasBox = document.createElement('div');
      canvasBox.className = 'chart-canvas-box';
      const canvas = document.createElement('canvas');
      canvasBox.appendChild(canvas);
      wrap.appendChild(canvasBox);

      pre.replaceWith(wrap);
      try { new Chart(canvas.getContext('2d'), _chartConfig(spec)); } catch (e) { console.error('[Chart]', e); }
    });
  }

  // Groups each ## (or #) heading and the elements that follow it into a
  // ".dash-section" card, so a dashboard-format reply reads as distinct
  // metric blocks instead of a flat wall of prose.
  function _wrapDashboardSections(container) {
    const nodes = Array.from(container.children);
    const frag = document.createDocumentFragment();
    let section = null;
    nodes.forEach(node => {
      if (/^H[12]$/.test(node.tagName)) {
        section = document.createElement('div');
        section.className = 'dash-section';
        frag.appendChild(section);
      }
      (section || frag).appendChild(node);
    });
    container.appendChild(frag);
  }

  function _esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function getContent(key) {
    return _registry[key]?.content || null;
  }

  return { open, openFromKey, setActive, closeTab, close, copyActive, register, getContent };
})();

// ── Settings ─────────────────────────────────────
const Settings = (() => {
  const DEFAULTS = { backendUrl: '' };
  let _s = { ...DEFAULTS };

  function load() {
    try { Object.assign(_s, JSON.parse(localStorage.getItem('ae_settings') || '{}')); } catch {}
    if (_s.backendUrl) _s.backendUrl = _s.backendUrl.replace(/\/$/, '');
    // Fall back to value injected by GitHub Actions (config.js → window.__BACKEND_URL__)
    if (!_s.backendUrl && window.__BACKEND_URL__) _s.backendUrl = window.__BACKEND_URL__;
    return _s;
  }
  function save(vals) {
    if (vals.backendUrl) vals.backendUrl = vals.backendUrl.replace(/\/$/, '');
    Object.assign(_s, vals);
    localStorage.setItem('ae_settings', JSON.stringify(_s));
  }
  function get(k) { return _s[k]; }
  // Returns an absolute URL when backendUrl is set, otherwise a relative path.
  function api(path) { return _s.backendUrl ? _s.backendUrl + path : path; }
  return { load, save, get, api, all: () => ({ ..._s }) };
})();

// ── Conversation store ───────────────────────────
const Store = (() => {
  const KEY = 'ae_conversations';
  let convs = [];
  let activeId = null;

  function load() {
    try { convs = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch { convs = []; }
    if (convs.length) activeId = convs[0].id;
  }
  function save() { localStorage.setItem(KEY, JSON.stringify(convs)); }

  function create(title = 'Nowa rozmowa') {
    const c = { id: Date.now().toString(), title, messages: [], createdAt: Date.now() };
    convs.unshift(c);
    activeId = c.id;
    save();
    return c;
  }

  function active() { return convs.find(c => c.id === activeId) || null; }

  function setActive(id) {
    activeId = id;
    return active();
  }

  function addMessage(role, content, format) {
    const c = active();
    if (!c) return;
    c.messages.push({ role, content, ts: Date.now(), format });
    if (c.messages.length === 2 && role === 'assistant') {
      c.title = c.messages[0].content.slice(0, 50).replace(/\n/g, ' ');
    }
    save();
  }

  function updateLastMessage(content, format) {
    const c = active();
    if (!c || !c.messages.length) return;
    const last = c.messages[c.messages.length - 1];
    if (last.role === 'assistant') {
      last.content = content;
      if (format) last.format = format;
    }
    save();
  }

  function deleteConv(id) {
    convs = convs.filter(c => c.id !== id);
    if (activeId === id) activeId = convs[0]?.id || null;
    save();
  }

  function clearAll() { convs = []; activeId = null; localStorage.removeItem(KEY); }

  return { load, create, active, setActive, addMessage, updateLastMessage, deleteConv, clearAll, all: () => convs };
})();

// ── Sidebar (FAQ + Documents) ─────────────────────
// Replaces the old chat-history list: surfaces the most common questions
// (aggregated across all conversations, so several phrasings of "new orders?"
// collapse into one entry) and every document/summary the assistant has
// generated, instead of making the user hunt through past threads.
const Sidebar = (() => {
  const STOPWORDS = new Set([
    'moje','moich','moja','mam','jakie','jakich','jaka','jaki','czy','o','w','na','z','ze','do','po',
    'dla','i','a','to','są','sa','jest','mi','mnie','mój','moim','ostatnie','ostatnich','pokaż','pokaz',
    'pokazać','pokazac','podaj','sprawdź','sprawdz','proszę','prosze','te','ten','ta','jak','ile','czym',
    'wszystkie','aktualne','aktywne','nowe','nowych','wygląda','wyglada','żeby','zeby','się','sie',
  ]);
  const PL_MAP = { ą:'a', ć:'c', ę:'e', ł:'l', ń:'n', ó:'o', ś:'s', ź:'z', ż:'z' };
  const SIM_THRESHOLD = 0.34;
  const MAX_FAQ = 8;
  const MAX_DOCS = 30;

  function _sigWords(text) {
    const norm = text.toLowerCase()
      .replace(/[ąćęłńóśźż]/g, c => PL_MAP[c])
      .replace(/[^a-z0-9\s]/g, ' ');
    return new Set(norm.split(/\s+/).filter(w => w.length > 2 && !STOPWORDS.has(w)));
  }

  function _jaccard(a, b) {
    if (!a.size && !b.size) return 1;
    let inter = 0;
    a.forEach(w => { if (b.has(w)) inter++; });
    const union = a.size + b.size - inter;
    return union === 0 ? 0 : inter / union;
  }

  // Groups differently-worded user questions into one aggregated entry (e.g.
  // "pokaż nowe zamówienia" / "czy są nowe zamówienia?" / "sprawdź zamówienia"
  // → one card) by clustering on shared significant words, then ranks
  // clusters by how often something in that cluster was asked.
  function _aggregateQuestions() {
    const clusters = [];
    Store.all().forEach(c => c.messages.forEach(m => {
      if (m.role !== 'user') return;
      const text = m.content.trim();
      if (!text || text.length > 200) return;
      const words = _sigWords(text);
      let best = null, bestSim = 0;
      clusters.forEach(cl => {
        const sim = _jaccard(words, cl.words);
        if (sim > bestSim) { bestSim = sim; best = cl; }
      });
      if (best && bestSim >= SIM_THRESHOLD) {
        best.count++;
        best.examples.set(text, (best.examples.get(text) || 0) + 1);
      } else {
        clusters.push({ words, count: 1, examples: new Map([[text, 1]]) });
      }
    }));
    return clusters
      .sort((a, b) => b.count - a.count)
      .slice(0, MAX_FAQ)
      .map(cl => {
        let rep = null, repCount = -1;
        cl.examples.forEach((cnt, text) => {
          if (cnt > repCount || (cnt === repCount && (!rep || text.length < rep.length))) {
            rep = text; repCount = cnt;
          }
        });
        return { text: rep, count: cl.count };
      });
  }

  // Mirrors Chat.buildBubble's isArtifact rule: a table only counts once it
  // has real data rows, document/dashboard replies always do.
  function _isDoc(content, format) {
    if (format === 'document' || format === 'dashboard') return true;
    if (format !== 'table') return false;
    const rows = content.split('\n').filter(l =>
      /^\s*\|.*\|\s*$/.test(l) && !/^\s*\|[\s:|-]+\|\s*$/.test(l));
    return rows.length > 1; // header row + at least one data row
  }

  function _docTitle(content, format) {
    const heading = content.match(/^#{1,3}\s+(.+)/m);
    if (heading) return heading[1].replace(/[*`_]/g, '').trim().slice(0, 70);
    const clean = content.replace(/[#*`_[\]|]/g, ' ').replace(/\s+/g, ' ').trim();
    return clean.slice(0, 70) || (format === 'dashboard' ? 'Analiza' : 'Dokument');
  }

  function _collectDocs() {
    const docs = [];
    Store.all().forEach(c => c.messages.forEach(m => {
      if (m.role !== 'assistant' || !m.content) return;
      const format = m.format || 'chat';
      if (!_isDoc(m.content, format)) return;
      docs.push({ ts: m.ts || 0, format, title: _docTitle(m.content, format), content: m.content });
    }));
    return docs.sort((a, b) => b.ts - a.ts).slice(0, MAX_DOCS);
  }

  let _faqCache = [];
  let _docsCache = [];

  function _renderFaq() {
    const el = document.getElementById('sidebar-faq');
    if (!el) return;
    _faqCache = _aggregateQuestions();
    el.innerHTML = _faqCache.length ? _faqCache.map((q, i) => `
      <button class="sidebar-list-item" onclick="Sidebar.ask(${i})" title="${escHtml(q.text)}">
        <span class="sli-icon">💡</span>
        <span class="sli-text">${escHtml(q.text)}</span>
        ${q.count > 1 ? `<span class="sli-count">${q.count}×</span>` : ''}
      </button>`).join('')
      : '<p class="sidebar-empty">Zadaj pytanie na czacie, aby zobaczyć tu najczęstsze tematy.</p>';
  }

  const DOC_ICON = { document: '📄', dashboard: '📊', table: '🗂️' };

  function _renderDocs() {
    const el = document.getElementById('sidebar-docs');
    if (!el) return;
    _docsCache = _collectDocs();
    el.innerHTML = _docsCache.length ? _docsCache.map((d, i) => `
      <button class="sidebar-list-item" onclick="Sidebar.openDoc(${i})" title="${escHtml(d.title)}">
        <span class="sli-icon">${DOC_ICON[d.format] || '📄'}</span>
        <span class="sli-text">${escHtml(d.title)}</span>
      </button>`).join('')
      : '<p class="sidebar-empty">Tu pojawią się dokumenty i podsumowania wygenerowane po zapytaniach.</p>';
  }

  function render() {
    _renderFaq();
    _renderDocs();
  }

  // Clicking a common question starts a fresh conversation with it, rather
  // than appending onto whatever thread happens to be active right now.
  function ask(i) {
    const q = _faqCache[i];
    if (!q) return;
    Chat.newConversation();
    Chat.send(q.text);
  }

  function openDoc(i) {
    const d = _docsCache[i];
    if (!d) return;
    DocViewer.open(d.title, d.content, d.format);
    document.getElementById('sidebar').classList.remove('open');
  }

  const TAB_KEY = 'ae_sidebar_tab';

  function switchTab(name) {
    const panels = { faq: 'sidebar-faq', docs: 'sidebar-docs' };
    Object.keys(panels).forEach(key => {
      document.getElementById(panels[key])?.classList.toggle('hidden', key !== name);
      const tab = document.getElementById(`sidebar-tab-${key}`);
      if (!tab) return;
      tab.classList.toggle('active', key === name);
      tab.setAttribute('aria-selected', key === name ? 'true' : 'false');
    });
    try { localStorage.setItem(TAB_KEY, name); } catch {}
  }

  function initTab() {
    const saved = (() => { try { return localStorage.getItem(TAB_KEY); } catch { return null; } })();
    switchTab(saved === 'docs' ? 'docs' : 'faq');
  }

  return { render, ask, openDoc, switchTab, initTab };
})();

// ── Backend API ──────────────────────────────────
const Backend = (() => {
  async function _doQuery(message, sessionId) {
    const res = await fetch(Settings.api('/query'), {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json', ...Auth.headers() },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        sender_id: 'web_user',
      })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    // "agent" is "<data_source>:<output_format>", e.g. "allegro_orders:table" —
    // the format half drives how the full-view doc viewer presents the reply.
    const format = (data.agent || '').split(':')[1] || 'chat';
    return { text: data.response, format };
  }

  // A couple of retries (with growing delay) covers the classic symptom of
  // racing a Cloud Run cold start (container scaled to zero, min-instances=0)
  // as well as a flaky mobile connection dropping a request outright.
  const _RETRY_DELAYS_MS = [1500, 3000];

  async function query(message, sessionId) {
    let lastErr;
    for (let attempt = 0; attempt <= _RETRY_DELAYS_MS.length; attempt++) {
      try {
        return await _doQuery(message, sessionId);
      } catch (err) {
        // fetch() itself rejects with a TypeError ("Load failed" / "Failed to
        // fetch") on network-level failures — as opposed to an HTTP error
        // response, which is thrown as a plain Error above. Only that class
        // of failure is worth retrying; a real HTTP error (4xx/5xx) would
        // just fail again the same way.
        if (!(err instanceof TypeError)) throw err;
        lastErr = err;
        if (attempt < _RETRY_DELAYS_MS.length) {
          await new Promise(r => setTimeout(r, _RETRY_DELAYS_MS[attempt]));
        }
      }
    }
    const netErr = new Error('Nie udało się połączyć z serwerem. Sprawdź internet i spróbuj ponownie.');
    netErr.isNetworkError = true;
    netErr.cause = lastErr;
    throw netErr;
  }
  return { query };
})();

// ── Web Push ─────────────────────────────────────
const WebPush = (() => {
  const SUB_KEY = 'ae_push_subscribed';

  function isSupported() {
    return 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window;
  }

  function _urlBase64ToUint8Array(b64) {
    const pad = '='.repeat((4 - b64.length % 4) % 4);
    const raw = atob((b64 + pad).replace(/-/g, '+').replace(/_/g, '/'));
    return Uint8Array.from([...raw].map(c => c.charCodeAt(0)));
  }

  async function subscribe() {
    if (!isSupported()) return false;
    try {
      // Safari/WebKit only honors Notification.requestPermission() when it runs
      // within the same synchronous gesture as the click — any await before it
      // (e.g. a network fetch) silently breaks the prompt on iOS. Ask first.
      if (Notification.permission !== 'granted') {
        const perm = await Notification.requestPermission();
        if (perm !== 'granted') return false;
      }

      const keyRes = await fetch(Settings.api('/push/vapid-public-key'), { credentials: 'include', headers: Auth.headers() });
      if (!keyRes.ok) return false;
      const { publicKey } = await keyRes.json();

      const reg = await navigator.serviceWorker.ready;
      let sub = await reg.pushManager.getSubscription();
      if (sub) {
        // A subscription is permanently bound to the VAPID key it was created
        // with — the browser won't let it silently follow a rotated server
        // key. Drop it and create a fresh one so it's guaranteed to match
        // the current public key (otherwise push fails with VapidPkHashMismatch
        // forever, and re-clicking "enable" looks like it does nothing).
        await fetch(Settings.api('/push/subscribe'), {
          method: 'DELETE',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json', ...Auth.headers() },
          body: JSON.stringify({ endpoint: sub.endpoint }),
        }).catch(() => {});
        await sub.unsubscribe();
      }
      sub = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: _urlBase64ToUint8Array(publicKey),
      });
      const subRes = await fetch(Settings.api('/push/subscribe'), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...Auth.headers() },
        body: JSON.stringify(sub.toJSON()),
      });
      if (!subRes.ok) {
        console.error('[WebPush] /push/subscribe failed, status:', subRes.status);
        return false;
      }
      localStorage.setItem(SUB_KEY, '1');
      return true;
    } catch (e) {
      console.error('[WebPush] subscribe error:', e);
      return false;
    }
  }

  // persist=true also stores this as an entry in the Notifications inbox (bell
  // icon panel) server-side, instead of injecting anything into the chat.
  async function sendNotification(title, body, persist, url, prompt) {
    const cleanBody = String(body).replace(/[#*`_~[\]]/g, '').replace(/\s+/g, ' ').trim().slice(0, 120);

    // Direct Notification — instant, for the current device (desktop/Android tab)
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        const n = new Notification(title, {
          body: cleanBody,
          icon: 'icons/icon-192.svg',
          tag: 'alleasystent-monitor',  // same tag so SW push replaces it silently
        });
        // Tapping the notification jumps straight into the chat question, same as the
        // OS-level push click handled by sw.js's notificationclick.
        if (prompt) n.onclick = () => { window.focus(); Chat.send(prompt); n.close(); };
      } catch {}
    }

    // Web Push — fans out to all subscribed devices (iOS PWA, other desktops, background tabs)
    // The SW shows a notification with the same tag, replacing the direct one on this device
    if (localStorage.getItem(SUB_KEY)) {
      const payload = { title, body: cleanBody, url: url ?? '/' };
      if (persist) payload.notify = true;
      if (prompt) payload.prompt = prompt;
      fetch(Settings.api('/push/notify'), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...Auth.headers() },
        body: JSON.stringify(payload),
      }).catch(() => {});
    }
  }

  async function checkPending() {
    // Retrieve and remove the oldest pending chat message from the server.
    // Called on app startup so devices that were offline during polling still see messages.
    try {
      const res = await fetch(Settings.api('/push/pending'), { credentials: 'include', headers: Auth.headers() });
      if (!res.ok) return null;
      const data = await res.json();
      return data.chatMessage || null;
    } catch { return null; }
  }

  async function init() {
    // Re-register subscription with backend on startup (token may have rotated)
    if (!isSupported() || !localStorage.getItem(SUB_KEY)) return;
    try {
      const reg = await navigator.serviceWorker.ready;
      const sub = await reg.pushManager.getSubscription();
      if (!sub) { localStorage.removeItem(SUB_KEY); return; }
      await fetch(Settings.api('/push/subscribe'), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...Auth.headers() },
        body: JSON.stringify(sub.toJSON()),
      }).catch(() => {});
    } catch {}
  }

  return { isSupported, subscribe, sendNotification, checkPending, init };
})();

// ── Order monitor ────────────────────────────────
const OrderMonitor = (() => {
  const ENABLED_KEY  = 'ae_monitor_enabled';
  const LAST_EVT_KEY = 'ae_monitor_last_event';
  let _timer = null;

  function isEnabled() { return localStorage.getItem(ENABLED_KEY) === '1'; }

  async function enable() {
    console.log('[OrderMonitor] enable() called');
    const pushOk = await WebPush.subscribe();
    console.log('[OrderMonitor] push subscribe result:', pushOk);
    localStorage.setItem(ENABLED_KEY, '1');
    fetch(Settings.api('/allegro/monitor/enable'), {
      method: 'POST', credentials: 'include', headers: Auth.headers(),
    }).catch(() => {});
    await _saveBaseline();
    if (_timer) clearInterval(_timer);
    _timer = setInterval(_check, 5 * 60 * 1000);
    console.log('[OrderMonitor] timer started, interval 5 min');
    if (pushOk) {
      UI.toast('✓ Monitoring zamówień włączony (co 5 minut)');
    } else {
      UI.toast('⚠️ Monitoring włączony, ale powiadomienia push nie działają — sprawdź uprawnienia powiadomień w przeglądarce/telefonie', 10000);
    }
    document.querySelectorAll('.btn-monitoring').forEach(btn => {
      btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring aktywny</span>';
    });
    return true;
  }

  async function _saveBaseline() {
    try {
      console.log('[OrderMonitor] saving baseline via /order-event-stats…');
      const res = await fetch(Settings.api('/allegro/order-event-stats'), { credentials: 'include', headers: Auth.headers() });
      console.log('[OrderMonitor] baseline HTTP', res.status);
      if (!res.ok) return;
      const data = await res.json();
      console.log('[OrderMonitor] baseline response:', JSON.stringify(data));
      if (data.latest_event_id) {
        localStorage.setItem(LAST_EVT_KEY, data.latest_event_id);
        console.log('[OrderMonitor] baseline saved, latest_event_id =', data.latest_event_id);
      } else {
        console.warn('[OrderMonitor] baseline response has no latest_event_id');
      }
    } catch (e) {
      console.error('[OrderMonitor] baseline fetch error:', e);
    }
  }

  function disable() {
    console.log('[OrderMonitor] disabled');
    localStorage.removeItem(ENABLED_KEY);
    if (_timer) { clearInterval(_timer); _timer = null; }
    fetch(Settings.api('/allegro/monitor/disable'), {
      method: 'POST', credentials: 'include', headers: Auth.headers(),
    }).catch(() => {});
  }

  async function _check() {
    const lastId = localStorage.getItem(LAST_EVT_KEY);
    console.log('[OrderMonitor] _check() lastId =', lastId, new Date().toISOString());
    if (!lastId) {
      console.warn('[OrderMonitor] no baseline — saving one and skipping this tick');
      await _saveBaseline();
      return;
    }
    try {
      const url = Settings.api(`/allegro/order-events?since=${encodeURIComponent(lastId)}`);
      const res = await fetch(url, { credentials: 'include', headers: Auth.headers() });
      console.log('[OrderMonitor] poll HTTP', res.status, 'url:', url);
      if (!res.ok) { console.error('[OrderMonitor] poll failed, status:', res.status); return; }
      const data = await res.json();
      console.log('[OrderMonitor] poll response:', JSON.stringify(data));
      if (data.last_event_id) localStorage.setItem(LAST_EVT_KEY, data.last_event_id);
      const count = (data.new_orders || []).length;
      if (count > 0) {
        const label = count === 1 ? 'zamówienie' : count < 5 ? 'zamówienia' : 'zamówień';
        const msg = `Masz ${count} nowe ${label} do realizacji!`;
        console.log('[OrderMonitor] NEW ORDERS DETECTED:', count, data.new_orders);
        // In-app toast only — no WebPush.sendNotification() here. The server-side
        // Cloud Run Job (services/order_monitor.py) detects the same event and
        // already sends the real push + inbox entry; also firing one from the
        // client would duplicate both, especially since this poll's local
        // baseline lags the job's and re-finds orders it already announced.
        UI.toast(`🛒 ${msg}`, 10000);
        Notifications.refresh();
      } else {
        console.log('[OrderMonitor] no new orders');
      }
    } catch (e) {
      console.error('[OrderMonitor] poll error:', e);
    }
  }

  function init(skipInitialCheck) {
    const enabled = isEnabled();
    const lastId  = localStorage.getItem(LAST_EVT_KEY);
    console.log('[OrderMonitor] init() enabled =', enabled, 'lastId =', lastId,
      'push =', !!localStorage.getItem('ae_push_subscribed'),
      'notif =', typeof Notification !== 'undefined' ? Notification.permission : 'unsupported');
    if (!enabled) return;
    // Auto-subscribe to Web Push if monitoring was enabled before VAPID was configured.
    // Works silently when Notification permission is already granted (no gesture needed).
    if (!localStorage.getItem('ae_push_subscribed') && typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      console.log('[OrderMonitor] init: attempting auto-subscribe to Web Push');
      WebPush.subscribe().then(ok => console.log('[OrderMonitor] auto-subscribe result:', ok)).catch(() => {});
    }
    if (_timer) clearInterval(_timer);
    // Skip the immediate check when the app was just opened by tapping a
    // notification — that notification IS the detection, so re-polling right
    // away just finds the same order again and fires a redundant duplicate.
    if (!skipInitialCheck) _check();
    _timer = setInterval(_check, 5 * 60 * 1000);
    console.log('[OrderMonitor] polling started');
  }

  return { isEnabled, enable, disable, init };
})();

// ── Invoice monitor ──────────────────────────────
const InvoiceMonitor = (() => {
  const ENABLED_KEY  = 'ae_invoice_monitor_enabled';
  const NOTIFIED_KEY = 'ae_invoice_notified_ids';
  let _timer = null;

  function isEnabled() { return localStorage.getItem(ENABLED_KEY) === '1'; }

  function _getNotified() {
    try { return new Set(JSON.parse(localStorage.getItem(NOTIFIED_KEY) || '[]')); }
    catch { return new Set(); }
  }

  function _saveNotified(set) {
    localStorage.setItem(NOTIFIED_KEY, JSON.stringify([...set].slice(-300)));
  }

  async function enable() {
    const pushOk = await WebPush.subscribe();
    localStorage.setItem(ENABLED_KEY, '1');
    fetch(Settings.api('/allegro/invoice-monitor/enable'), {
      method: 'POST', credentials: 'include', headers: Auth.headers(),
    }).catch(() => {});
    _startPolling(); // first check notifies about ALL currently pending invoices
    if (pushOk) {
      UI.toast('✓ Monitoring faktur włączony (co 15 minut)');
    } else {
      UI.toast('⚠️ Monitoring włączony, ale powiadomienia push nie działają — sprawdź uprawnienia powiadomień w przeglądarce/telefonie', 10000);
    }
    document.querySelectorAll('.btn-invoice-monitoring').forEach(btn => {
      btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring faktur aktywny</span>';
    });
    return true;
  }

  function disable() {
    localStorage.removeItem(ENABLED_KEY);
    if (_timer) { clearInterval(_timer); _timer = null; }
    fetch(Settings.api('/allegro/invoice-monitor/disable'), {
      method: 'POST', credentials: 'include', headers: Auth.headers(),
    }).catch(() => {});
  }

  async function _check() {
    try {
      const res = await fetch(Settings.api('/allegro/pending-invoices'), { credentials: 'include', headers: Auth.headers() });
      if (!res.ok) return;
      const data = await res.json();
      const orders = data.orders || [];
      if (orders.length === 0) return;

      const notified = _getNotified();
      const newOnes = orders.filter(o => !notified.has(o.order_id));
      if (newOnes.length === 0) return;

      newOnes.forEach(o => notified.add(o.order_id));
      _saveNotified(notified);
      const count = newOnes.length;
      const label = count === 1 ? 'zamówienie wymaga' : count < 5 ? 'zamówienia wymagają' : 'zamówień wymaga';
      const msg = `${count} ${label} wystawienia faktury VAT.`;
      const prompt = count === 1
        ? 'Podaj mi szczegóły zamówienia, które wymaga wystawienia faktury VAT.'
        : `Podaj mi szczegóły ${count} zamówień, które wymagają wystawienia faktury VAT.`;
      UI.toast(`🧾 ${msg}`, 10000);
      WebPush.sendNotification('AllEasystent — Faktura VAT!', msg, true, '/?open=notifications', prompt);
      Notifications.refresh();
    } catch (e) {}
  }

  function _startPolling(skipInitialCheck) {
    if (_timer) clearInterval(_timer);
    // Skip the immediate check when the app was just opened by tapping a
    // notification — that notification IS the detection, so re-polling right
    // away just finds the same invoice again and fires a redundant duplicate.
    if (!skipInitialCheck) _check();
    _timer = setInterval(_check, 15 * 60 * 1000);
  }

  function init(skipInitialCheck) {
    if (!isEnabled()) return;
    if (!localStorage.getItem('ae_push_subscribed') && typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      WebPush.subscribe().catch(() => {});
    }
    _startPolling(skipInitialCheck);
  }

  return { isEnabled, enable, disable, init };
})();

// ── Message monitor ───────────────────────────────
const MessageMonitor = (() => {
  const ENABLED_KEY  = 'ae_message_monitor_enabled';
  const NOTIFIED_KEY = 'ae_message_notified_ids';
  let _timer = null;

  function isEnabled() { return localStorage.getItem(ENABLED_KEY) === '1'; }

  function _getNotified() {
    try { return new Set(JSON.parse(localStorage.getItem(NOTIFIED_KEY) || '[]')); }
    catch { return new Set(); }
  }

  function _saveNotified(set) {
    localStorage.setItem(NOTIFIED_KEY, JSON.stringify([...set].slice(-300)));
  }

  async function enable() {
    const pushOk = await WebPush.subscribe();
    localStorage.setItem(ENABLED_KEY, '1');
    fetch(Settings.api('/allegro/message-monitor/enable'), {
      method: 'POST', credentials: 'include', headers: Auth.headers(),
    }).catch(() => {});
    _startPolling(); // first check notifies about ALL currently unread threads
    if (pushOk) {
      UI.toast('✓ Monitoring wiadomości włączony (co 10 minut)');
    } else {
      UI.toast('⚠️ Monitoring włączony, ale powiadomienia push nie działają — sprawdź uprawnienia powiadomień w przeglądarce/telefonie', 10000);
    }
    document.querySelectorAll('.btn-message-monitoring').forEach(btn => {
      btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring wiadomości aktywny</span>';
    });
    return true;
  }

  function disable() {
    localStorage.removeItem(ENABLED_KEY);
    if (_timer) { clearInterval(_timer); _timer = null; }
    fetch(Settings.api('/allegro/message-monitor/disable'), {
      method: 'POST', credentials: 'include', headers: Auth.headers(),
    }).catch(() => {});
  }

  async function _check() {
    try {
      const res = await fetch(Settings.api('/allegro/unread-messages'), { credentials: 'include', headers: Auth.headers() });
      if (!res.ok) return;
      const data = await res.json();
      const threads = data.threads || [];
      if (threads.length === 0) return;

      const notified = _getNotified();
      const newOnes = threads.filter(t => !notified.has(t.thread_id));
      if (newOnes.length === 0) return;

      newOnes.forEach(t => notified.add(t.thread_id));
      _saveNotified(notified);
      const count = newOnes.length;
      const msg = count === 1
        ? '1 nowa nieprzeczytana wiadomość od kupującego.'
        : `${count} nowych nieprzeczytanych wiadomości od kupujących.`;
      const prompt = count === 1
        ? 'Pokaż mi tę nową wiadomość od kupującego.'
        : `Pokaż mi te ${count} nowe wiadomości od kupujących.`;
      UI.toast(`💬 ${msg}`, 10000);
      WebPush.sendNotification('AllEasystent — Nowa wiadomość!', msg, true, '/?open=notifications', prompt);
      Notifications.refresh();
    } catch (e) {}
  }

  function _startPolling(skipInitialCheck) {
    if (_timer) clearInterval(_timer);
    // Skip the immediate check when the app was just opened by tapping a
    // notification — that notification IS the detection, so re-polling right
    // away just finds the same thread again and fires a redundant duplicate.
    if (!skipInitialCheck) _check();
    _timer = setInterval(_check, 10 * 60 * 1000);
  }

  function init(skipInitialCheck) {
    if (!isEnabled()) return;
    if (!localStorage.getItem('ae_push_subscribed') && typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      WebPush.subscribe().catch(() => {});
    }
    _startPolling(skipInitialCheck);
  }

  return { isEnabled, enable, disable, init };
})();

// ── Returns & complaints monitor ─────────────────
// Unlike Invoice/MessageMonitor above, detection itself runs entirely
// server-side (the same Cloud Run Job cadence as OrderMonitor, see
// services/return_complaint_monitor.py) — there's no lightweight "since X"
// endpoint to poll from the tab, so this object only handles the toggle
// (push subscribe + enable/disable) and lets the server-sent push notify.
const ReturnsMonitor = (() => {
  const ENABLED_KEY = 'ae_returns_monitor_enabled';

  function isEnabled() { return localStorage.getItem(ENABLED_KEY) === '1'; }

  async function enable() {
    const pushOk = await WebPush.subscribe();
    localStorage.setItem(ENABLED_KEY, '1');
    fetch(Settings.api('/allegro/returns-monitor/enable'), {
      method: 'POST', credentials: 'include', headers: Auth.headers(),
    }).catch(() => {});
    if (pushOk) {
      UI.toast('✓ Monitoring zwrotów i reklamacji włączony');
    } else {
      UI.toast('⚠️ Monitoring włączony, ale powiadomienia push nie działają — sprawdź uprawnienia powiadomień w przeglądarce/telefonie', 10000);
    }
    document.querySelectorAll('.btn-returns-monitoring').forEach(btn => {
      btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring zwrotów i reklamacji aktywny</span>';
    });
    return true;
  }

  function disable() {
    localStorage.removeItem(ENABLED_KEY);
    fetch(Settings.api('/allegro/returns-monitor/disable'), {
      method: 'POST', credentials: 'include', headers: Auth.headers(),
    }).catch(() => {});
  }

  function init() {
    if (!isEnabled()) return;
    if (!localStorage.getItem('ae_push_subscribed') && typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      WebPush.subscribe().catch(() => {});
    }
  }

  return { isEnabled, enable, disable, init };
})();

// ── Notifications (bell icon panel) ──────────────
const Notifications = (() => {
  let _items = [];
  const CACHE_KEY = 'ae_notifications_cache';

  // Entries can land in `_items` from several independent sources — the cached
  // snapshot, the server's refresh(), a tapped notification's URL params, a
  // still-pending OS tray notification — each trusting its own idea of "goes on
  // top". Re-sorting by created_at on every mutation, instead of relying on each
  // source to insert in the right relative position, is what keeps the list
  // correct newest-first regardless of which sources have (or haven't yet) run.
  function _sortByNewest(items) {
    return [...items].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  }

  function _loadCache() {
    try {
      const cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null');
      if (!cached) return false;
      _items = _sortByNewest(cached.items || []);
      _renderBadge(cached.unread_count || 0);
      _render();
      return true;
    } catch { return false; }
  }

  function _saveCache(data) {
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify({
        items: data.items || [], unread_count: data.unread_count || 0,
      }));
    } catch {}
  }

  function _esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function _timeAgo(iso) {
    const min = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
    if (min < 1) return 'przed chwilą';
    if (min < 60) return `${min} min temu`;
    const h = Math.floor(min / 60);
    if (h < 24) return `${h} godz. temu`;
    return `${Math.floor(h / 24)} dni temu`;
  }

  function _renderBadge(count) {
    document.querySelectorAll('.notif-badge').forEach(badge => {
      if (count > 0) {
        badge.textContent = count > 99 ? '99+' : String(count);
        badge.classList.remove('hidden');
      } else {
        badge.classList.add('hidden');
      }
    });
  }

  function _render() {
    const list = document.getElementById('notifications-list');
    if (!list) return;
    if (_items.length === 0) {
      list.innerHTML = '<div class="notif-empty muted">Brak powiadomień.</div>';
      return;
    }
    list.innerHTML = _items.map(n => `
      <div class="notif-item${n.read ? '' : ' unread'}${n.prompt ? ' has-action' : ''}"
           ${n.prompt ? `onclick="Notifications.activate('${n.id}')"` : ''}>
        <div class="notif-title">${_esc(n.title)}</div>
        <div class="notif-body">${_esc(n.body)}</div>
        <div class="notif-time">${_timeAgo(n.created_at)}</div>
      </div>
    `).join('');
  }

  async function refresh() {
    try {
      const res = await fetch(Settings.api('/notifications'), { credentials: 'include', headers: Auth.headers() });
      if (!res.ok) return;
      const data = await res.json();
      _items = _sortByNewest(data.items || []);
      _renderBadge(data.unread_count || 0);
      _render();
      _saveCache(data);
    } catch (e) {
      console.error('[Notifications] refresh error:', e);
      // Offline / network failure — fall back to the last cached snapshot
      // instead of leaving the panel empty.
      if (!_items.length) _loadCache();
    }
  }

  async function open() {
    document.getElementById('notifications-overlay').classList.remove('hidden');
    document.getElementById('notifications-panel').classList.remove('hidden');
    await refresh();
    if (_items.some(n => !n.read)) {
      fetch(Settings.api('/notifications/mark-read'), {
        method: 'POST', credentials: 'include', headers: Auth.headers(),
      }).then(() => _renderBadge(0)).catch(() => {});
    }
  }

  function close() {
    document.getElementById('notifications-overlay').classList.add('hidden');
    document.getElementById('notifications-panel').classList.add('hidden');
  }

  // Tapping a notification with a ready-made question fires it straight into the
  // chat instead of just closing the panel — same behavior as tapping the OS push.
  function activate(id) {
    const n = _items.find(i => i.id === id);
    close();
    if (n?.prompt) Chat.send(n.prompt);
  }

  function init() {
    _loadCache();  // instant paint from last known state, then refresh from network
    _consumePending();
    refresh();
  }

  // Paint a notification straight from the OS push's own payload (carried in via
  // launch URL params — see sw.js's notificationclick) instead of waiting for the
  // background refresh() to round-trip to the server and confirm it exists. That
  // refresh() still runs right after and replaces `_items` with the server's
  // authoritative list — the same entry is already there by the time it lands,
  // since the backend persists it before sending the push — so this is purely a
  // "show it now" shortcut, not a second source of truth.
  function applyPending(entry) {
    if (!entry?.id || _items.some(i => i.id === entry.id)) return;
    _items = _sortByNewest([entry, ..._items]);
    const unread = _items.filter(i => !i.read).length;
    _renderBadge(unread);
    _render();
    _saveCache({ items: _items, unread_count: unread });
  }

  // Covers the other launch path: the user saw the OS notification but opened
  // the app itself (icon/app-switcher) instead of tapping it, so notificationclick
  // never ran and no payload arrived via URL params. The notification the SW
  // showed is still sitting in the OS tray/notification-center with its own data
  // attached (sw.js's showNotification sets `data`) — read it back directly via
  // the Notifications API instead of waiting on refresh()'s round-trip to learn
  // the same thing. Tapped notifications are excluded automatically: sw.js's
  // notificationclick already closes them before the app even loads.
  async function _consumePending() {
    if (!('serviceWorker' in navigator)) return;
    try {
      const reg = await navigator.serviceWorker.ready;
      const notifs = await reg.getNotifications({ tag: 'alleasystent-monitor' });
      // Order doesn't matter here — applyPending() re-sorts the whole list by
      // created_at on every call, so it lands correctly regardless of the order
      // getNotifications() (which makes no ordering guarantee) hands them back in.
      notifs.forEach(n => {
        if (n.data?.id) applyPending({ ...n.data, read: false });
        n.close();
      });
    } catch {}
  }

  return { open, close, refresh, activate, init, applyPending };
})();

// ── UI helpers ───────────────────────────────────
const UI = (() => {
  let _toastT = null;

  function toast(msg, ms = 2500) {
    clearTimeout(_toastT);
    const el = document.getElementById('toast');
    el.textContent = msg; el.classList.remove('hidden');
    _toastT = setTimeout(() => el.classList.add('hidden'), ms);
  }

  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
  }

  function openSettings() {
    document.getElementById('settings-overlay').classList.remove('hidden');
    document.getElementById('settings-panel').classList.remove('hidden');
    document.getElementById('set-toggle-orders').checked = OrderMonitor.isEnabled();
    document.getElementById('set-toggle-invoices').checked = InvoiceMonitor.isEnabled();
    document.getElementById('set-toggle-messages').checked = MessageMonitor.isEnabled();
    document.getElementById('set-toggle-returns').checked = ReturnsMonitor.isEnabled();
    updateVersionInfo();
  }

  function closeSettings() {
    document.getElementById('settings-overlay').classList.add('hidden');
    document.getElementById('settings-panel').classList.add('hidden');
  }

  function toggleOrderMonitoring(on) {
    if (on) OrderMonitor.enable(); else OrderMonitor.disable();
  }

  function toggleInvoiceMonitoring(on) {
    if (on) InvoiceMonitor.enable(); else InvoiceMonitor.disable();
  }

  function toggleMessageMonitoring(on) {
    if (on) MessageMonitor.enable(); else MessageMonitor.disable();
  }

  function toggleReturnsMonitoring(on) {
    if (on) ReturnsMonitor.enable(); else ReturnsMonitor.disable();
  }

  function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
  }

  function exportChat() {
    const c = Store.active();
    if (!c || !c.messages.length) { toast('Brak wiadomości do eksportu'); return; }
    const text = c.messages.map(m => `[${m.role === 'user' ? 'Ty' : 'AllEasystent'}]\n${m.content}`).join('\n\n---\n\n');
    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `alleasystent-${new Date().toISOString().slice(0,10)}.txt`;
    a.click();
  }

  function clearAllHistory() {
    if (!confirm('Usunąć całą historię rozmów?')) return;
    Store.clearAll();
    Chat.newConversation();
    closeSettings();
    toast('Historia usunięta');
  }

  return {
    toast, autoResize, openSettings, closeSettings, toggleSidebar, exportChat, clearAllHistory,
    toggleOrderMonitoring, toggleInvoiceMonitoring, toggleMessageMonitoring, toggleReturnsMonitoring,
  };
})();

// ── Chat engine ──────────────────────────────────
const Chat = (() => {
  let _waiting = false;
  let _welcomeEl = null;  // persistent ref so GC never collects the node

  function renderSidebar() {
    Sidebar.render();
  }

  function renderMessages() {
    const c = Store.active();
    const container = document.getElementById('messages');
    // Always resolve via cache — getElementById returns null after the node
    // has been removed from DOM by a previous container.innerHTML = ''
    if (!_welcomeEl) _welcomeEl = document.getElementById('welcome');
    container.innerHTML = '';

    if (!c || !c.messages.length) {
      if (_welcomeEl) container.appendChild(_welcomeEl);
      return;
    }

    c.messages.forEach((m, i) => {
      const el = buildBubble(m.role, m.content, m.ts, i, m.format);
      container.appendChild(el);
      _applyMonitoringState(el);
    });
    scrollBottom();
  }

  function _applyMonitoringState(bubbleEl) {
    const inner = bubbleEl.querySelector('.msg-bubble');
    if (!inner) return;
    // Re-auth prompt from the Allegro agent — must go through startAllegroLogin()
    // (fetches a signed backend auth URL) rather than a plain href, since a bare
    // "/allegro/login" link resolves against the wrong origin on the split
    // GitHub Pages / Cloud Run deployment and 404s.
    if (inner.innerHTML.includes('[ALLEGRO_LOGIN_BTN]')) {
      inner.innerHTML = inner.innerHTML.replace('[ALLEGRO_LOGIN_BTN]',
        '<button class="btn-monitoring" onclick="AllegroAuth.start(this)">➡ Zaloguj się przez Allegro</button>');
    }
    // Fallback for old text markers (LLM paraphrasing)
    if (inner.innerHTML.includes('[ORDER_MONITORING_BTN]')) {
      inner.innerHTML = inner.innerHTML.replace('[ORDER_MONITORING_BTN]',
        '<button class="btn-monitoring" onclick="OrderMonitor.enable()">🔔 Włącz monitoring zamówień</button>');
    }
    if (inner.innerHTML.includes('[INVOICE_MONITORING_BTN]')) {
      inner.innerHTML = inner.innerHTML.replace('[INVOICE_MONITORING_BTN]',
        '<button class="btn-invoice-monitoring" onclick="InvoiceMonitor.enable()">🧾 Włącz monitoring faktur</button>');
    }
    if (inner.innerHTML.includes('[MESSAGE_MONITORING_BTN]')) {
      inner.innerHTML = inner.innerHTML.replace('[MESSAGE_MONITORING_BTN]',
        '<button class="btn-message-monitoring" onclick="MessageMonitor.enable()">💬 Włącz monitoring wiadomości</button>');
    }
    if (inner.innerHTML.includes('[RETURNS_MONITORING_BTN]')) {
      inner.innerHTML = inner.innerHTML.replace('[RETURNS_MONITORING_BTN]',
        '<button class="btn-returns-monitoring" onclick="ReturnsMonitor.enable()">↩️ Włącz monitoring zwrotów i reklamacji</button>');
    }
    // Replace enable-buttons with active badge if monitoring is already on.
    // Only touches actual "enable" buttons — get_new_orders' status block already
    // renders a "disable" button of its own when monitoring is on, and that one
    // must stay a clickable button, not get flattened into a static badge.
    if (OrderMonitor.isEnabled()) {
      inner.querySelectorAll('.btn-monitoring').forEach(btn => {
        if (btn.getAttribute('onclick')?.includes('OrderMonitor.enable')) {
          btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring zamówień aktywny</span>';
        }
      });
    }
    if (InvoiceMonitor.isEnabled()) {
      inner.querySelectorAll('.btn-invoice-monitoring').forEach(btn => {
        if (btn.getAttribute('onclick')?.includes('InvoiceMonitor.enable')) {
          btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring faktur aktywny</span>';
        }
      });
    }
    if (MessageMonitor.isEnabled()) {
      inner.querySelectorAll('.btn-message-monitoring').forEach(btn => {
        if (btn.getAttribute('onclick')?.includes('MessageMonitor.enable')) {
          btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring wiadomości aktywny</span>';
        }
      });
    }
    if (ReturnsMonitor.isEnabled()) {
      inner.querySelectorAll('.btn-returns-monitoring').forEach(btn => {
        if (btn.getAttribute('onclick')?.includes('ReturnsMonitor.enable')) {
          btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring zwrotów i reklamacji aktywny</span>';
        }
      });
    }
  }

  // Finds the markdown table (if any) in a bot reply and counts its data rows.
  // Shared by _tablePreview (chat-bubble summary) and buildBubble (decides
  // whether a table-format reply is substantial enough to be treated as an
  // artifact — an empty/no-result table must NOT trigger the doc viewer).
  function _tableStats(content) {
    const lines = content.split('\n');
    const isTableLine = (l) => /^\s*\|.*\|\s*$/.test(l);
    const isSepLine = (l) => /^\s*\|[\s:|-]+\|\s*$/.test(l);
    let lastTableLineIdx = -1;
    let dataRows = 0;
    lines.forEach((l, i) => {
      if (isTableLine(l)) {
        lastTableLineIdx = i;
        if (!isSepLine(l)) dataRows++;
      }
    });
    if (lastTableLineIdx === -1) return { hasTable: false, dataRows: 0, lines, lastTableLineIdx };
    return { hasTable: true, dataRows: Math.max(dataRows - 1, 0), lines, lastTableLineIdx };
  }

  // Table-format responses put the markdown table first and any summary
  // sentence last — slicing the first 220 raw chars just shows garbled
  // "| Zamówienie | Kupujący | ... | :…" table syntax. Prefer the trailing
  // summary sentence, or a row-count label, over dumping the table itself.
  function _tablePreview(content) {
    const stats = _tableStats(content);
    if (!stats.hasTable) return null;
    const trailing = stats.lines.slice(stats.lastTableLineIdx + 1).join(' ')
      .replace(/[#*`_[\]]/g, '').trim();
    if (trailing) {
      return trailing.slice(0, 220) + (trailing.length > 220 ? '…' : '');
    }
    const noun = stats.dataRows === 1 ? 'wiersz' : 'wierszy';
    return `📊 Tabela — ${stats.dataRows} ${noun}. Kliknij „Pełny widok”, aby zobaczyć szczegóły.`;
  }

  // Action buttons (monitoring toggle etc.) are always appended at the very end of
  // bot content — pull them out so they never get sliced apart by preview truncation.
  function _extractTrailingHtml(content) {
    const m = content.match(/<button[\s\S]*$/);
    if (!m) return { text: content, html: null };
    return { text: content.slice(0, m.index).trimEnd(), html: m[0] };
  }

  // Some document-format replies lead with a ```summary fenced block — a short
  // bulleted summary meant for the chat bubble (see get_order_details in
  // agents/allegro/allegro_agent.py), with the fuller detail (products,
  // billing, ...) reserved for the doc viewer. Pulls it out and returns the
  // remaining text separately so it isn't shown twice / rendered as a raw code block.
  function _extractSummaryBlock(content) {
    const m = content.match(/```summary\r?\n([\s\S]*?)```[ \t]*\r?\n?/);
    if (!m) return { summary: null, rest: content };
    const summary = m[1].trim();
    const rest = (content.slice(0, m.index) + content.slice(m.index + m[0].length)).replace(/^\s+/, '');
    return { summary, rest };
  }

  // Formats that always belong in the full-window doc viewer, the same way
  // Claude pops a description into an artifact instead of dumping it in the
  // chat bubble: tables get a chat summary + full table doc, documents and
  // dashboards open straight away (see finalizeWaitingBubble's autoOpen).
  const _ARTIFACT_FORMATS = new Set(['table', 'document', 'dashboard']);

  // Short, human preview for document/dashboard replies: leading heading (if
  // any) + the first line of body text — no raw markdown/table dump.
  function _docPreview(bodyText, format) {
    if (format !== 'document' && format !== 'dashboard') return null;
    const heading = bodyText.match(/^#{1,2}\s+(.+)/m);
    const title = heading ? heading[1].replace(/[*`_]/g, '').trim() : null;
    const rest = heading ? bodyText.slice(bodyText.indexOf(heading[0]) + heading[0].length) : bodyText;
    const lead = rest.replace(/^#+\s*/mg, '').replace(/[*`_[\]]/g, '').replace(/\s+/g, ' ').trim();
    const leadShort = lead.slice(0, 160) + (lead.length > 160 ? '…' : '');
    const icon = format === 'dashboard' ? '📊' : '📄';
    if (title) return `${icon} ${title}${leadShort ? ' — ' + leadShort : ''}`;
    return `${icon} ${leadShort || (format === 'dashboard' ? 'Analiza gotowa.' : 'Dokument gotowy.')}`;
  }

  function buildBubble(role, content, ts, index, format, autoOpen) {
    const isUser = role === 'user';
    const { text: bodyTextRaw, html: trailingHtml } = isUser
      ? { text: content, html: null }
      : _extractTrailingHtml(content);
    // Pull out a leading ```summary block, if any — it's the chat-bubble
    // preview; the doc viewer only gets the remaining detail content, so the
    // summary isn't shown twice or rendered as a raw code block there.
    const { summary: summaryBlock, rest: bodyText } = isUser
      ? { summary: null, rest: bodyTextRaw }
      : _extractSummaryBlock(bodyTextRaw);
    // A "table" reply only counts as an artifact when it actually contains rows —
    // an empty/no-results table (e.g. "brak nowych wiadomości") must render as a
    // normal short chat reply, not get hidden behind a doc-viewer link.
    const tableStats = !isUser ? _tableStats(bodyText) : null;
    const isArtifact = !isUser && _ARTIFACT_FORMATS.has(format)
      && (format !== 'table' || (tableStats.hasTable && tableStats.dataRows > 0));
    // Plain "chat"/"action" replies are never truncated, no matter how long —
    // only table/document/dashboard artifacts get the compact-preview +
    // doc-viewer treatment. Truncating ordinary text answers just to make the
    // user click "Zobacz pełną odpowiedź" was the exact thing this was meant to fix.
    const isLong = !isUser && (summaryBlock !== null || (isArtifact && bodyText.length > 150));
    const div = document.createElement('div');
    div.className = `msg msg-${isUser ? 'user' : 'bot'}`;
    div.dataset.index = index ?? '';

    const avatar = isUser ? '👤' : '🛒';
    const time = ts ? new Date(ts).toLocaleTimeString('pl-PL', { hour: '2-digit', minute: '2-digit' }) : '';

    // Register long bot responses so "Pełny widok" button can re-open the doc viewer.
    // Doc content = detail body (summary block stripped) + any trailing button HTML.
    const docContent = trailingHtml ? bodyText + '\n\n' + trailingHtml : bodyText;
    const docKey = isLong ? DocViewer.register(docContent, format) : null;

    // Long responses: show a compact preview in the bubble — full content is in the doc viewer
    let bubbleHtml;
    if (isLong) {
      let previewHtml;
      if (summaryBlock !== null) {
        // Model-authored bulleted summary — render as real markdown, not a
        // single truncated line.
        previewHtml = `<div style="font-size:.88rem;margin:0 0 .5rem">${renderMarkdown(summaryBlock)}</div>`;
      } else {
        const tablePreview = _tablePreview(bodyText);
        let previewShort;
        if (tablePreview !== null) {
          previewShort = escHtml(tablePreview);
        } else {
          const docPreview = _docPreview(bodyText, format);
          if (docPreview !== null) {
            previewShort = escHtml(docPreview);
          } else {
            const preview = bodyText.replace(/^#+\s*/mg, '').replace(/[*`_[\]]/g, '').trim();
            previewShort = escHtml(preview.slice(0, 220)) + (preview.length > 220 ? '…' : '');
          }
        }
        previewHtml = `<p style="color:var(--muted);font-size:.88rem;margin:0 0 .5rem">${previewShort}</p>`;
      }
      bubbleHtml = previewHtml +
        `<a href="javascript:void(0)" onclick="DocViewer.openFromKey(${docKey})" ` +
        `style="display:inline-block;font-size:.85rem;font-weight:600;color:var(--accent);text-decoration:none">` +
        `📄 Zobacz pełną odpowiedź →</a>`;
    } else {
      bubbleHtml = isUser ? escHtml(content).replace(/\n/g, '<br>') : renderMarkdown(bodyText);
    }
    // Buttons render for real (clickable), always outside the truncated preview
    if (trailingHtml) bubbleHtml += renderMarkdown(trailingHtml);

    div.innerHTML = `
      <div class="msg-avatar">${avatar}</div>
      <div class="msg-content">
        <div class="msg-bubble"${docKey !== null ? ` data-doc-key="${docKey}"` : ''}>${bubbleHtml}</div>
        <div class="msg-actions">
          <button class="msg-act-btn" onclick="Chat.copyMessage(this)" title="Kopiuj">📋 Kopiuj</button>
          ${!isUser ? `<button class="msg-act-btn" onclick="Chat.regenerate()" title="Generuj ponownie">↺ Nowa odpowiedź</button>` : ''}
          ${docKey !== null ? `<button class="msg-act-btn msg-act-doc" onclick="DocViewer.openFromKey(${docKey})">📄 Pełny widok</button>` : ''}
        </div>
        ${time ? `<span class="msg-time">${time}</span>` : ''}
      </div>`;

    // Auto-open the full-window doc viewer for fresh table/document/dashboard
    // replies — mirrors Claude popping an artifact open instead of making the
    // user hunt for a "show more" link. Only requested for live new replies
    // (see finalizeWaitingBubble); history reloads never pass autoOpen.
    if (autoOpen && docKey !== null && isArtifact) {
      DocViewer.openFromKey(docKey);
    }
    return div;
  }

  function appendBotBubble() {
    const container = document.getElementById('messages');
    const welcome = document.getElementById('welcome');
    if (container.contains(welcome)) container.removeChild(welcome);

    const div = document.createElement('div');
    div.className = 'msg msg-bot';
    div.id = 'waiting-bubble';
    div.innerHTML = `
      <div class="msg-avatar">🛒</div>
      <div class="msg-content">
        <div class="msg-bubble" id="waiting-content">
          <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>
      </div>`;
    container.appendChild(div);
    scrollBottom();
    return document.getElementById('waiting-content');
  }

  function finalizeWaitingBubble(fullText, ts, format) {
    const bubble = document.getElementById('waiting-bubble');
    if (!bubble) return;
    const idx = Store.active()?.messages.length - 1;
    const replacement = buildBubble('assistant', fullText, ts, idx, format, true);
    bubble.replaceWith(replacement);
    _applyMonitoringState(replacement);
    if (typeof hljs !== 'undefined') {
      replacement.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
    }
  }

  function scrollBottom() {
    const el = document.getElementById('messages');
    el.scrollTop = el.scrollHeight;
  }

  async function send(text) {
    if (_waiting) return;
    const input = document.getElementById('user-input');
    const msgText = (text || input.value).trim();
    if (!msgText) return;

    if (!Store.active()) Store.create();
    Store.addMessage('user', msgText);
    input.value = ''; input.style.height = 'auto';

    const container = document.getElementById('messages');
    const welcome = document.getElementById('welcome');
    if (container.contains(welcome)) container.removeChild(welcome);
    const msgs = Store.active().messages;
    container.appendChild(buildBubble('user', msgText, msgs[msgs.length-1].ts, msgs.length-1));
    scrollBottom();
    renderSidebar();

    await _dispatch(msgText);
  }

  // Runs the query and renders the reply. Shared by send() (user bubble
  // already appended) and retryLast() (reuses the user bubble already on
  // screen instead of appending a duplicate one).
  async function _dispatch(msgText) {
    _waiting = true;
    document.getElementById('btn-send').disabled = true;
    appendBotBubble();

    // Long queries (big report, cold-start backend) can leave the typing
    // dots spinning with no feedback — after 10s, let the user know we're
    // still working on it instead of leaving them guessing.
    const slowTimer = setTimeout(() => {
      const contentEl = document.getElementById('waiting-content');
      if (!contentEl) return;
      contentEl.insertAdjacentHTML('beforeend',
        `<p style="margin:.5rem 0 0;color:var(--muted);font-size:.85rem">⏳ Pobieram informacje, daj mi jeszcze chwilkę…</p>`);
    }, 10000);

    const sessionId = Store.active().id;
    const ts = Date.now();
    let fullText = '';
    let fullFormat = 'chat';
    let isError = false;

    try {
      Store.addMessage('assistant', '');
      const result = await Backend.query(msgText, sessionId);
      fullText = result.text;
      fullFormat = result.format;
      Store.updateLastMessage(fullText, fullFormat);
    } catch (err) {
      isError = true;
      if (err.isNetworkError) {
        // Backend.query already retried a few times — this only fires once
        // those are exhausted, so tell the user plainly and let them retry
        // by hand instead of silently failing.
        fullText = `⚠️ **Problem z połączeniem.** ${err.message}\n\n` +
          `<button class="btn-monitoring" style="background:#ef4444" onclick="Chat.retryLast()">🔄 Zapytaj ponownie</button>`;
      } else {
        fullText = `**Błąd:** ${err.message}`;
      }
      const contentEl = document.getElementById('waiting-content');
      if (contentEl) contentEl.innerHTML = `<span style="color:#fca5a5">${escHtml(err.message)}</span>`;
      Store.updateLastMessage(fullText);
      UI.toast(`Błąd: ${err.message}`, 5000);
    } finally {
      clearTimeout(slowTimer);
      _waiting = false;
      document.getElementById('btn-send').disabled = false;
      finalizeWaitingBubble(fullText, ts, fullFormat);
      renderSidebar();
      if (typeof hljs !== 'undefined') {
        document.querySelectorAll('#messages pre code').forEach(b => hljs.highlightElement(b));
      }
      // Notify if the tab was in the background when the response arrived
      if (document.hidden && fullText && !isError) {
        WebPush.sendNotification('AllEasystent', fullText);
      }
    }
  }

  // "Zapytaj ponownie" button on a failed-connection bubble: drop that error
  // message and re-issue the same last user prompt, without re-adding a
  // duplicate user bubble (unlike regenerate(), which is meant to re-ask).
  function retryLast() {
    if (_waiting) return;
    const c = Store.active();
    if (!c || !c.messages.length) return;
    const last = c.messages[c.messages.length - 1];
    if (last.role !== 'assistant') return;
    c.messages.pop();
    localStorage.setItem('ae_conversations', JSON.stringify(Store.all()));
    renderMessages();
    const lastUser = [...c.messages].reverse().find(m => m.role === 'user');
    if (lastUser) _dispatch(lastUser.content);
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  }

  function sendSuggestion(btn) { send(btn.textContent); }

  function newConversation() {
    Store.create();
    renderSidebar();
    renderMessages();
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('user-input').focus();
  }

  function loadConversation(id) {
    Store.setActive(id);
    renderSidebar();
    renderMessages();
    document.getElementById('sidebar').classList.remove('open');
  }

  function deleteConversation(id) {
    Store.deleteConv(id);
    renderSidebar();
    renderMessages();
  }

  function copyMessage(btn) {
    const bubble = btn.closest('.msg-content').querySelector('.msg-bubble');
    const docKey = bubble.dataset.docKey;
    const text = docKey ? (DocViewer.getContent(parseInt(docKey)) || bubble.innerText) : bubble.innerText;
    navigator.clipboard?.writeText(text).then(() => UI.toast('Skopiowano ✓')).catch(() => UI.toast('Błąd kopiowania'));
  }

  async function regenerate() {
    const c = Store.active();
    if (!c || c.messages.length < 2) return;
    c.messages.pop();
    localStorage.setItem('ae_conversations', JSON.stringify(Store.all()));
    renderMessages();
    const lastUser = [...c.messages].reverse().find(m => m.role === 'user');
    if (lastUser) await send(lastUser.content);
  }

  return { send, handleKey, sendSuggestion, newConversation, loadConversation, deleteConversation, copyMessage, regenerate, retryLast };
})();

// ── Allegro OAuth login ──────────────────────────────────────────────────
// Shared by the login-overlay button AND by [ALLEGRO_LOGIN_BTN] buttons the
// chat agent injects when it needs re-auth mid-conversation. Both must go
// through the backend for a signed auth URL and redirect there — a bare
// href like "/allegro/login" is wrong on the GitHub Pages / Cloud Run split
// deployment, where that path resolves against the frontend's own origin
// (which doesn't have the route) instead of the backend's.
//
// Cache the OAuth URL in localStorage (20-minute TTL) so the button
// redirects instantly even on a cold-start — the prefetch fires immediately
// on page load to wake the container, and the result is cached for the next
// time the user visits. The HMAC-signed state has no server-side expiry so
// caching for a few minutes is safe.
const _AUTH_URL_LS_KEY = 'ae_allegro_auth_url';
const _AUTH_URL_TTL_MS = 20 * 60 * 1000; // 20 min — state is stateless HMAC, safe to cache longer

function _getCachedAllegroAuthUrl() {
  try {
    const raw = localStorage.getItem(_AUTH_URL_LS_KEY);
    if (!raw) return null;
    const { url, ts } = JSON.parse(raw);
    if (Date.now() - ts > _AUTH_URL_TTL_MS) { localStorage.removeItem(_AUTH_URL_LS_KEY); return null; }
    return url;
  } catch { return null; }
}

function _setCachedAllegroAuthUrl(url) {
  try { localStorage.setItem(_AUTH_URL_LS_KEY, JSON.stringify({ url, ts: Date.now() })); } catch {}
}

// Seed the promise from cache so first click is instant; prefetch always
// fires to wake the container and refresh the cached URL for next time.
let _allegroAuthUrlPromise = Promise.resolve(_getCachedAllegroAuthUrl());

function _prefetchAllegroAuthUrl() {
  _allegroAuthUrlPromise = fetch(Settings.api('/allegro/auth-url'), { credentials: 'include' })
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(d => { _setCachedAllegroAuthUrl(d.auth_url); return d.auth_url; })
    .catch(() => _getCachedAllegroAuthUrl()); // fall back to stale cache on network error
}

let _allegroLoginInProgress = false;
async function startAllegroLogin(btn) {
  if (_allegroLoginInProgress) return;
  _allegroLoginInProgress = true;

  // Fast path: cached URL → fire wake-up ping + redirect immediately (no spinner)
  const cachedUrl = _getCachedAllegroAuthUrl();
  if (cachedUrl) {
    wakeContainer(); // start warming the container while user is on Allegro's page
    window.location.href = cachedUrl;
    return;
  }

  // Slow path: still waiting for backend
  const origHTML = btn ? btn.innerHTML : null;
  if (btn) {
    btn.innerHTML = '⏳ Łączenie…';
    btn.style.opacity = '0.65';
    btn.style.pointerEvents = 'none';
  }

  try {
    const auth_url = await _allegroAuthUrlPromise;
    if (!auth_url) throw new Error('no url');
    window.location.href = auth_url;
    // don't restore — page is navigating away
  } catch {
    if (btn) {
      btn.innerHTML = origHTML;
      btn.style.opacity = '';
      btn.style.pointerEvents = '';
    }
    _allegroLoginInProgress = false;
    _prefetchAllegroAuthUrl(); // refresh so user can retry
    UI.toast('Błąd połączenia z backendem', 'error');
  }
}

// Exposed for [ALLEGRO_LOGIN_BTN] buttons injected into chat messages (see
// _applyMonitoringState), same pattern as OrderMonitor/InvoiceMonitor/MessageMonitor.
window.AllegroAuth = { start: startAllegroLogin };

// ── Boot ─────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  OfflineBanner.init();
  Settings.load();
  Store.load();
  updateVersionInfo();

  _prefetchAllegroAuthUrl();

  const loginBtn = document.getElementById('login-btn');
  if (loginBtn) {
    loginBtn.removeAttribute('href');
    loginBtn.addEventListener('click', (e) => {
      e.preventDefault();
      startAllegroLogin(loginBtn);
    });
  }

  const logoutLink = document.getElementById('logout-link');
  if (logoutLink) {
    logoutLink.href = Settings.api('/auth/logout');
    logoutLink.addEventListener('click', () => Auth.clearToken());
  }

  // Handle Allegro OAuth callback: read ?code= and ?state= from URL.
  const _urlParams = new URLSearchParams(window.location.search);
  const oauthCode = _urlParams.get('code') || sessionStorage.getItem('ae_oauth_code');
  const oauthState = _urlParams.get('state') || sessionStorage.getItem('ae_oauth_state');
  sessionStorage.removeItem('ae_oauth_code');
  sessionStorage.removeItem('ae_oauth_state');
  if (oauthCode && oauthState) {
    // Show spinner immediately — user already returned from Allegro, hide the button.
    const _loginAction = document.getElementById('login-action');
    const _loginSpinner = document.getElementById('login-spinner');
    document.getElementById('login-overlay').style.display = 'flex';
    if (_loginAction) _loginAction.style.display = 'none';
    if (_loginSpinner) _loginSpinner.style.display = '';
    // Clean URL immediately so refresh doesn't re-trigger exchange
    window.history.replaceState({}, '', window.location.pathname);
    try {
      const res = await fetch(Settings.api('/allegro/exchange'), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: oauthCode, state: oauthState }),
      });
      if (res.ok) {
        // Store the JWT in localStorage so Safari (which blocks cross-site
        // Set-Cookie) can still authenticate subsequent requests via Bearer token.
        const data = await res.json().catch(() => ({}));
        if (data.token) Auth.setToken(data.token);
        // Cache Allegro token expiry so the UI can inform the user
        if (data.allegro_expires_at) {
          try { localStorage.setItem('ae_allegro_expires', data.allegro_expires_at); } catch {}
        }
      } else {
        const err = await res.json().catch(() => ({}));
        const msg = err.detail || res.status;
        console.error('[allegro/exchange] failed:', res.status, err);
        if (_loginSpinner) _loginSpinner.style.display = 'none';
        if (_loginAction) _loginAction.style.display = '';
        alert('Błąd logowania przez Allegro (' + res.status + '): ' + msg);
      }
    } catch (e) {
      console.error('[allegro/exchange] network error:', e);
      if (_loginSpinner) _loginSpinner.style.display = 'none';
      if (_loginAction) _loginAction.style.display = '';
      alert('Błąd połączenia podczas logowania: ' + e.message);
    }
  }

  // Check authentication first — show login overlay if not logged in
  const authed = await checkAuth();
  if (!authed) return;

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').then(reg => {
      // Listen before triggering a check, so a fast-resolving update() can't
      // fire 'updatefound' before anyone is listening for it.
      // When a new SW is found, show the update banner once it finishes
      // installing. Reload happens only from the banner (click or its own
      // auto-reload timeout) — that's the single path that triggers a reload.
      reg.addEventListener('updatefound', () => {
        reg.installing?.addEventListener('statechange', e => {
          if (e.target.state === 'installed' && navigator.serviceWorker.controller) {
            AppUpdater.showBanner();
          }
        });
      });
      // Some browsers (Safari especially) are lazy about spontaneously
      // re-checking sw.js for changes — force a check on every page load.
      reg.update().catch(() => {});
    }).catch(() => {});
  }

  document.addEventListener('click', e => {
    const sidebar = document.getElementById('sidebar');
    if (sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target.id !== 'btn-sidebar-toggle') {
      sidebar.classList.remove('open');
    }
  });

  const convs = Store.all();
  if (!convs.length) Store.create('Nowa rozmowa');

  Sidebar.initTab();
  Sidebar.render();

  const active = Store.active();
  if (!active || !active.messages.length) {
    document.getElementById('messages').appendChild(document.getElementById('welcome'));
    document.getElementById('welcome').classList.remove('hidden');
  } else {
    Chat.loadConversation(active.id);
  }

  document.getElementById('user-input').focus();

  // Re-register push subscription with backend (token may have rotated)
  WebPush.init();

  // Check for pending chat messages stored on server (sent while this device was offline)
  WebPush.checkPending().then(text => {
    if (!text) return;
    try {
      if (!Store.active()) Store.create();
      const conv = Store.active();
      if (!conv) return;
      const isDup = conv.messages.some(m => m.content === text);
      if (isDup) return;
      conv.messages.push({ role: 'assistant', content: text, ts: Date.now() });
      localStorage.setItem('ae_conversations', JSON.stringify(Store.all()));
      setTimeout(() => Chat.loadConversation(conv.id), 0);
    } catch {}
  }).catch(() => {});

  // Init monitors AFTER full UI setup so chat injection finds a ready DOM
  const _startParams = new URLSearchParams(location.search);
  const _cameFromNotification = _startParams.get('open') === 'notifications';
  OrderMonitor.init(_cameFromNotification);
  InvoiceMonitor.init(_cameFromNotification);
  MessageMonitor.init(_cameFromNotification);
  ReturnsMonitor.init();

  Notifications.init();
  // Tapping a system push notification opens straight into the Notifications
  // panel — and skips the monitors' immediate re-check above, since the
  // notification IS the detection; re-polling on arrival would just find the
  // same order/invoice again and fire a redundant duplicate.
  if (_cameFromNotification) {
    // The tapped notification's own title/body/prompt travel in as launch params
    // (see sw.js) — paint it into the inbox right away instead of leaving the
    // panel showing stale/empty state until Notifications.open()'s refresh()
    // reaches the server.
    const nid = _startParams.get('nid');
    if (nid) {
      Notifications.applyPending({
        id: nid,
        title: _startParams.get('ntitle') || '',
        body: _startParams.get('nbody') || '',
        prompt: _startParams.get('nprompt') || null,
        created_at: _startParams.get('ncreated') || new Date().toISOString(),
        read: false,
      });
    }
    Notifications.open();
    history.replaceState(null, '', location.pathname);
  }
});
