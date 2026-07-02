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
const AppUpdater = (() => {
  let _knownInstance = null;
  let _bannerShown = false;

  function _showBanner() {
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
    // Auto-reload after 10 s if user hasn't clicked
    setTimeout(() => AppUpdater.reload(), 10000);
  }

  function check(headers) {
    const inst = headers?.get?.('X-Server-Instance');
    if (!inst) return;
    if (!_knownInstance) { _knownInstance = inst; return; }
    if (_knownInstance !== inst) _showBanner();
  }

  function reload() {
    // Tell SW to activate immediately, then reload
    navigator.serviceWorker?.getRegistration?.()?.then?.(reg => {
      if (reg?.waiting) reg.waiting.postMessage({ type: 'SKIP_WAITING' });
      else window.location.reload();
    }) ?? window.location.reload();
  }

  return { check, reload, showBanner: _showBanner };
})();

// ── Auth check ────────────────────────────────────
async function checkAuth() {
  try {
    const res = await fetch(Settings.api('/auth/me'), { credentials: 'include' });
    AppUpdater.check(res.headers);
    if (res.status === 401) {
      document.getElementById('login-overlay').style.display = 'flex';
      return false;
    }
    const user = await res.json();
    window._currentUser = user;
    document.getElementById('login-overlay').style.display = 'none';
    document.getElementById('app').style.display = '';
    const userEl = document.getElementById('user-info');
    if (userEl) {
      userEl.innerHTML = `<span style="font-size:1.1rem">🛒</span> <span style="overflow:hidden;text-overflow:ellipsis;font-weight:500">${user.name}</span>`;
    }
    return true;
  } catch (e) {
    document.getElementById('login-overlay').style.display = 'flex';
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
    const res = await fetch(Settings.api('/query'), {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        sender_id: 'web_user',
      })
    });
    AppUpdater.check(res.headers);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    return data.response;
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
      const keyRes = await fetch(Settings.api('/push/vapid-public-key'), { credentials: 'include' });
      if (!keyRes.ok) return false;
      const { publicKey } = await keyRes.json();

      const reg = await navigator.serviceWorker.ready;
      let sub = await reg.pushManager.getSubscription();
      if (!sub) {
        const perm = await Notification.requestPermission();
        if (perm !== 'granted') return false;
        sub = await reg.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey: _urlBase64ToUint8Array(publicKey),
        });
      }
      await fetch(Settings.api('/push/subscribe'), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sub.toJSON()),
      });
      localStorage.setItem(SUB_KEY, '1');
      return true;
    } catch (e) {
      console.error('[WebPush] subscribe error:', e);
      return false;
    }
  }

  // chatText is the full markdown chat message — backend stores it in Redis so
  // other devices (e.g. iOS PWA) can retrieve it via /push/pending on startup.
  async function sendNotification(title, body, chatText, url) {
    const cleanBody = String(body).replace(/[#*`_~[\]]/g, '').replace(/\s+/g, ' ').trim().slice(0, 120);

    // Direct Notification — instant, for the current device (desktop/Android tab)
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        new Notification(title, {
          body: cleanBody,
          icon: 'icons/icon-192.svg',
          tag: 'alleasystent-monitor',  // same tag so SW push replaces it silently
        });
      } catch {}
    }

    // Web Push — fans out to all subscribed devices (iOS PWA, other desktops, background tabs)
    // The SW shows a notification with the same tag, replacing the direct one on this device
    if (localStorage.getItem(SUB_KEY)) {
      const payload = { title, body: cleanBody, url: url ?? '/' };
      if (chatText) payload.chatMessage = chatText;
      fetch(Settings.api('/push/notify'), {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).catch(() => {});
    }
  }

  async function checkPending() {
    // Retrieve and remove the oldest pending chat message from the server.
    // Called on app startup so devices that were offline during polling still see messages.
    try {
      const res = await fetch(Settings.api('/push/pending'), { credentials: 'include' });
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
      await fetch('/push/subscribe', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
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
    await WebPush.subscribe();
    localStorage.setItem(ENABLED_KEY, '1');
    await _saveBaseline();
    if (_timer) clearInterval(_timer);
    _timer = setInterval(_check, 5 * 60 * 1000);
    console.log('[OrderMonitor] timer started, interval 5 min');
    UI.toast('✓ Monitoring zamówień włączony (co 5 minut)');
    document.querySelectorAll('.btn-monitoring').forEach(btn => {
      btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring aktywny</span>';
    });
    return true;
  }

  async function _saveBaseline() {
    try {
      console.log('[OrderMonitor] saving baseline via /order-event-stats…');
      const res = await fetch(Settings.api('/allegro/order-event-stats'), { credentials: 'include' });
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
      const res = await fetch(url, { credentials: 'include' });
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
        UI.toast(`🛒 ${msg}`, 10000);
        // Await so the chat text is ready (and localStorage written) before the push fires.
        // Backend stores the text in Redis so other devices receive it on startup.
        const chatText = await _injectChatMessage(data.new_orders);
        WebPush.sendNotification('AllEasystent — Nowe zamówienie!', msg, chatText);
      } else {
        console.log('[OrderMonitor] no new orders');
      }
    } catch (e) {
      console.error('[OrderMonitor] poll error:', e);
    }
  }

  function init() {
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
    _check();
    _timer = setInterval(_check, 5 * 60 * 1000);
    console.log('[OrderMonitor] polling started');
  }

  async function _injectChatMessage(orders) {
    try {
      if (!Store.active()) Chat.newConversation();
      const targetConvId = Store.active().id;

      const details = await Promise.all(
        orders.map(o => o.order_id
          ? fetch(Settings.api(`/allegro/orders/${encodeURIComponent(o.order_id)}`), { credentials: 'include' })
              .then(r => r.ok ? r.json() : null)
              .catch(() => null)
          : Promise.resolve(null)
        )
      );

      const header = orders.length === 1
        ? '🛒 **Nowe zamówienie do realizacji**'
        : `🛒 **${orders.length} nowe zamówienia do realizacji**`;

      const blocks = orders.map((o, i) => {
        const d = details[i];
        if (!d) return `**${String(o.order_id || '').slice(0, 8)}…** — brak szczegółów`;
        const total = `${Number(d.total_price).toFixed(2)} zł`;
        const itemLines = (d.items || []).map(it => `  • ${it.name} ×${it.quantity}`).join('\n');
        return `👤 **${d.buyer_login}** · ${total}\n📦 ${d.delivery_method}\n${itemLines}`;
      }).join('\n\n---\n\n');

      const text = `${header}\n\n${blocks}`;

      const conv = Store.all().find(c => c.id === targetConvId);
      if (conv) {
        conv.messages.push({ role: 'assistant', content: text, ts: Date.now() });
        localStorage.setItem('ae_conversations', JSON.stringify(Store.all()));
        setTimeout(() => Chat.loadConversation(targetConvId), 0);
      }
      return text;  // returned so caller can pass it to WebPush.sendNotification
    } catch (e) {
      console.error('[OrderMonitor] chat inject error:', e);
      return null;
    }
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
    await WebPush.subscribe();
    localStorage.setItem(ENABLED_KEY, '1');
    _startPolling(); // first check notifies about ALL currently pending invoices
    UI.toast('✓ Monitoring faktur włączony (co 15 minut)');
    document.querySelectorAll('.btn-invoice-monitoring').forEach(btn => {
      btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring faktur aktywny</span>';
    });
    return true;
  }

  function disable() {
    localStorage.removeItem(ENABLED_KEY);
    if (_timer) { clearInterval(_timer); _timer = null; }
  }

  async function _check() {
    try {
      const res = await fetch(Settings.api('/allegro/pending-invoices'), { credentials: 'include' });
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
      UI.toast(`🧾 ${msg}`, 10000);
      const chatText = _injectChatMessage(newOnes);
      WebPush.sendNotification('AllEasystent — Faktura VAT!', msg, chatText);
    } catch (e) {}
  }

  function _injectChatMessage(orders) {
    try {
      if (!Store.active()) Chat.newConversation();
      const targetConvId = Store.active().id;
      const lines = orders.map(o => {
        const buyer = o.buyer || '—';
        const amount = o.total != null ? ` · ${Number(o.total).toFixed(2)} zł` : '';
        return `- **${String(o.order_id).slice(0, 8)}…** · ${buyer}${amount}`;
      }).join('\n');
      const noun = orders.length === 1 ? 'zamówienie wymagające' : `${orders.length} zamówień wymagających`;
      const text = `🧾 **Monitoring faktur** — wykryto ${noun} faktury VAT:\n\n${lines}\n\nPamiętaj o wystawieniu faktury dla każdego z nich.`;
      const conv = Store.all().find(c => c.id === targetConvId);
      if (conv) {
        conv.messages.push({ role: 'assistant', content: text, ts: Date.now() });
        localStorage.setItem('ae_conversations', JSON.stringify(Store.all()));
        setTimeout(() => Chat.loadConversation(targetConvId), 0);
      }
      return text;
    } catch { return null; }
  }

  function _startPolling() {
    if (_timer) clearInterval(_timer);
    _check();
    _timer = setInterval(_check, 15 * 60 * 1000);
  }

  function init() {
    if (!isEnabled()) return;
    if (!localStorage.getItem('ae_push_subscribed') && typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      WebPush.subscribe().catch(() => {});
    }
    _startPolling();
  }

  return { isEnabled, enable, disable, init };
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
  let _welcomeEl = null;  // persistent ref so GC never collects the node

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
    // Always resolve via cache — getElementById returns null after the node
    // has been removed from DOM by a previous container.innerHTML = ''
    if (!_welcomeEl) _welcomeEl = document.getElementById('welcome');
    container.innerHTML = '';

    if (!c || !c.messages.length) {
      if (_welcomeEl) container.appendChild(_welcomeEl);
      return;
    }

    c.messages.forEach((m, i) => {
      const el = buildBubble(m.role, m.content, m.ts, i);
      container.appendChild(el);
      _applyMonitoringState(el);
    });
    scrollBottom();
  }

  function _applyMonitoringState(bubbleEl) {
    const inner = bubbleEl.querySelector('.msg-bubble');
    if (!inner) return;
    // Fallback for old text markers (LLM paraphrasing)
    if (inner.innerHTML.includes('[ORDER_MONITORING_BTN]')) {
      inner.innerHTML = inner.innerHTML.replace('[ORDER_MONITORING_BTN]',
        '<button class="btn-monitoring" onclick="OrderMonitor.enable()">🔔 Włącz monitoring zamówień</button>');
    }
    if (inner.innerHTML.includes('[INVOICE_MONITORING_BTN]')) {
      inner.innerHTML = inner.innerHTML.replace('[INVOICE_MONITORING_BTN]',
        '<button class="btn-invoice-monitoring" onclick="InvoiceMonitor.enable()">🧾 Włącz monitoring faktur</button>');
    }
    // Replace enable-buttons with active badge if monitoring is already on
    if (OrderMonitor.isEnabled()) {
      inner.querySelectorAll('.btn-monitoring').forEach(btn => {
        btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring zamówień aktywny</span>';
      });
    }
    if (InvoiceMonitor.isEnabled()) {
      inner.querySelectorAll('.btn-invoice-monitoring').forEach(btn => {
        btn.outerHTML = '<span class="monitoring-badge">✓ Monitoring faktur aktywny</span>';
      });
    }
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
    _applyMonitoringState(replacement);
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
      // Notify if the tab was in the background when the response arrived
      if (document.hidden && fullText && !fullText.startsWith('**Błąd:**')) {
        WebPush.sendNotification('AllEasystent', fullText);
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

  // Point login / logout links at the backend (handles split deployment where
  // frontend lives on GitHub Pages and backend on Cloud Run).
  const loginBtn = document.getElementById('login-btn');
  if (loginBtn) loginBtn.href = Settings.api('/allegro/login');
  const logoutLink = document.getElementById('logout-link');
  if (logoutLink) logoutLink.href = Settings.api('/auth/logout');

  // Check authentication first — show login overlay if not logged in
  const authed = await checkAuth();
  if (!authed) return;

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').then(reg => {
      // When a new SW is found, show update banner once it finishes installing
      reg.addEventListener('updatefound', () => {
        reg.installing?.addEventListener('statechange', e => {
          if (e.target.state === 'installed' && navigator.serviceWorker.controller) {
            AppUpdater.showBanner();
          }
        });
      });
    }).catch(() => {});
    // After SKIP_WAITING the controller changes — reload to serve new assets
    let _reloading = false;
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (!_reloading) { _reloading = true; window.location.reload(); }
    });
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
  OrderMonitor.init();
  InvoiceMonitor.init();
});
