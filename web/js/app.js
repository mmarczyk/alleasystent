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

// ── Auth check ────────────────────────────────────
async function checkAuth() {
  try {
    const res = await fetch('/auth/me', { credentials: 'include' });
    if (res.status === 401) {
      document.getElementById('login-overlay').style.display = 'flex';
      return false;
    }
    const user = await res.json();
    // Store user info, update UI
    window._currentUser = user;
    // Hide overlay if somehow visible
    const overlay = document.getElementById('login-overlay');
    if (overlay) overlay.style.display = 'none';
    // Show user info in sidebar if element exists
    const userEl = document.getElementById('user-info');
    if (userEl) {
      userEl.innerHTML = `<img src="${user.picture}" style="width:28px;height:28px;border-radius:50%;flex-shrink:0"> <span style="overflow:hidden;text-overflow:ellipsis">${user.name}</span>`;
    }
    return true;
  } catch (e) {
    return false;
  }
}

// ── Settings ─────────────────────────────────────
const Settings = (() => {
  const DEFAULTS = { backendUrl: '' };
  let _s = { ...DEFAULTS };

  function load() {
    try { Object.assign(_s, JSON.parse(localStorage.getItem('ae_settings') || '{}')); } catch {}
    if (_s.backendUrl) _s.backendUrl = _s.backendUrl.replace(/\/$/, '');
    return _s;
  }
  function save(vals) {
    if (vals.backendUrl) vals.backendUrl = vals.backendUrl.replace(/\/$/, '');
    Object.assign(_s, vals);
    localStorage.setItem('ae_settings', JSON.stringify(_s));
  }
  function get(k) { return _s[k]; }
  return { load, save, get, all: () => ({ ..._s }) };
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

  function addMessage(role, content) {
    const c = active();
    if (!c) return;
    c.messages.push({ role, content, ts: Date.now() });
    if (c.messages.length === 2 && role === 'assistant') {
      c.title = c.messages[0].content.slice(0, 50).replace(/\n/g, ' ');
    }
    save();
  }

  function updateLastMessage(content) {
    const c = active();
    if (!c || !c.messages.length) return;
    const last = c.messages[c.messages.length - 1];
    if (last.role === 'assistant') last.content = content;
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

// ── Backend API ──────────────────────────────────
const Backend = (() => {
  async function query(message, sessionId) {
    const backendUrl = Settings.get('backendUrl');
    const url = backendUrl ? `${backendUrl}/query` : '/query';
    const res = await fetch(url, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
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
    return data.response;
  }
  return { query };
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
    document.getElementById('set-backend-url').value = Settings.get('backendUrl');
  }

  function closeSettings() {
    document.getElementById('settings-overlay').classList.add('hidden');
    document.getElementById('settings-panel').classList.add('hidden');
  }

  function saveSettings() {
    Settings.save({ backendUrl: document.getElementById('set-backend-url').value.trim() });
    closeSettings();
    toast('Ustawienia zapisane ✓');
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

  return { toast, autoResize, openSettings, closeSettings, saveSettings, toggleSidebar, exportChat, clearAllHistory };
})();

// ── Chat engine ──────────────────────────────────
const Chat = (() => {
  let _waiting = false;

  function renderSidebar() {
    const list = document.getElementById('sidebar-history');
    const convs = Store.all();
    const active = Store.active();
    list.innerHTML = convs.length ? convs.map(c => `
      <div class="history-item ${c.id === active?.id ? 'active' : ''}" onclick="Chat.loadConversation('${c.id}')">
        <span class="hi-icon">💬</span>
        <span style="overflow:hidden;text-overflow:ellipsis;flex:1">${escHtml(c.title)}</span>
        <button class="hi-del" onclick="event.stopPropagation();Chat.deleteConversation('${c.id}')" title="Usuń">✕</button>
      </div>`).join('')
    : '<p style="color:var(--muted);font-size:.8rem;padding:.5rem .75rem">Brak rozmów</p>';
  }

  function renderMessages() {
    const c = Store.active();
    const container = document.getElementById('messages');
    const welcome = document.getElementById('welcome');
    container.innerHTML = '';

    if (!c || !c.messages.length) {
      container.appendChild(welcome);
      return;
    }

    c.messages.forEach((m, i) => {
      container.appendChild(buildBubble(m.role, m.content, m.ts, i));
    });
    scrollBottom();
  }

  function buildBubble(role, content, ts, index) {
    const isUser = role === 'user';
    const div = document.createElement('div');
    div.className = `msg msg-${isUser ? 'user' : 'bot'}`;
    div.dataset.index = index ?? '';

    const avatar = isUser ? '👤' : '🛒';
    const html = isUser ? escHtml(content).replace(/\n/g, '<br>') : renderMarkdown(content);
    const time = ts ? new Date(ts).toLocaleTimeString('pl-PL', { hour: '2-digit', minute: '2-digit' }) : '';

    div.innerHTML = `
      <div class="msg-avatar">${avatar}</div>
      <div class="msg-content">
        <div class="msg-bubble">${html}</div>
        <div class="msg-actions">
          <button class="msg-act-btn" onclick="Chat.copyMessage(this)" title="Kopiuj">📋 Kopiuj</button>
          ${!isUser ? `<button class="msg-act-btn" onclick="Chat.regenerate()" title="Generuj ponownie">↺ Nowa odpowiedź</button>` : ''}
        </div>
        ${time ? `<span class="msg-time">${time}</span>` : ''}
      </div>`;
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

  function finalizeWaitingBubble(fullText, ts) {
    const bubble = document.getElementById('waiting-bubble');
    if (!bubble) return;
    const idx = Store.active()?.messages.length - 1;
    const replacement = buildBubble('assistant', fullText, ts, idx);
    bubble.replaceWith(replacement);
    if (typeof hljs !== 'undefined') {
      replacement.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
    }
  }

  function renderMarkdown(text) {
    if (typeof marked === 'undefined') return escHtml(text).replace(/\n/g, '<br>');
    return marked.parse(text);
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
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

    const backendUrl = Settings.get('backendUrl');

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

    _waiting = true;
    document.getElementById('btn-send').disabled = true;
    appendBotBubble();

    const sessionId = Store.active().id;
    const ts = Date.now();
    let fullText = '';

    try {
      Store.addMessage('assistant', '');
      fullText = await Backend.query(msgText, sessionId);
      Store.updateLastMessage(fullText);
    } catch (err) {
      fullText = `**Błąd:** ${err.message}`;
      const contentEl = document.getElementById('waiting-content');
      if (contentEl) contentEl.innerHTML = `<span style="color:#fca5a5">${escHtml(err.message)}</span>`;
      Store.updateLastMessage(fullText);
      UI.toast(`Błąd: ${err.message}`, 5000);
    } finally {
      _waiting = false;
      document.getElementById('btn-send').disabled = false;
      finalizeWaitingBubble(fullText, ts);
      renderSidebar();
      if (typeof hljs !== 'undefined') {
        document.querySelectorAll('#messages pre code').forEach(b => hljs.highlightElement(b));
      }
    }
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
    navigator.clipboard?.writeText(bubble.innerText).then(() => UI.toast('Skopiowano ✓')).catch(() => UI.toast('Błąd kopiowania'));
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

  return { send, handleKey, sendSuggestion, newConversation, loadConversation, deleteConversation, copyMessage, regenerate };
})();

// ── Boot ─────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  Settings.load();
  Store.load();

  // Check authentication first — show login overlay if not logged in
  const authed = await checkAuth();
  if (!authed) return;

  if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

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

  (() => {
    const list = document.getElementById('sidebar-history');
    const all = Store.all();
    const active = Store.active();
    list.innerHTML = all.length ? all.map(c => `
      <div class="history-item ${c.id === active?.id ? 'active' : ''}" onclick="Chat.loadConversation('${c.id}')">
        <span class="hi-icon">💬</span>
        <span style="overflow:hidden;text-overflow:ellipsis;flex:1">${c.title}</span>
        <button class="hi-del" onclick="event.stopPropagation();Chat.deleteConversation('${c.id}')" title="Usuń">✕</button>
      </div>`).join('')
    : '<p style="color:var(--muted);font-size:.8rem;padding:.5rem .75rem">Brak rozmów</p>';
  })();

  const active = Store.active();
  if (!active || !active.messages.length) {
    document.getElementById('messages').appendChild(document.getElementById('welcome'));
    document.getElementById('welcome').classList.remove('hidden');
  } else {
    Chat.loadConversation(active.id);
  }

  document.getElementById('user-input').focus();
});
