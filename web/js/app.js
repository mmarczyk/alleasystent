/* ═══════════════════════════════════════════════════
   AllEasystent Chat UI — main controller
   ═══════════════════════════════════════════════════ */

// ── System prompt ────────────────────────────────
const SYSTEM_PROMPTS = {
  pl: `Jesteś AllEasystent — ekspert AI dla polskich właścicieli sklepów internetowych, specjalizujący się w platformie Allegro.

Pomagasz w:
- **Oferty i produkty**: tytuły SEO, opisy, parametry, zdjęcia, kategorie
- **Sprzedaż**: strategie cenowe, promocje, Allegro Ads, analiza konkurencji
- **Obsługa klienta**: szablony wiadomości, odpowiedzi na pytania, reklamacje i zwroty
- **Logistyka**: metody wysyłki, Allegro Smart, pakowanie, zarządzanie magazynem
- **Marketing**: kampanie, social media, cross-selling, upselling
- **Analityka**: interpretacja statystyk, prognozowanie, optymalizacja konwersji
- **Prawo e-commerce**: regulaminy, RODO, prawa konsumenta

Odpowiadasz po polsku, profesjonalnie i konkretnie. Używasz formatowania markdown — nagłówków, list i pogrubień — gdy poprawia to czytelność. Podajesz gotowe do użycia szablony i przykłady. Jeśli pytanie dotyczy czegoś poza e-commerce, możesz odpowiedzieć, ale zaznacz że to nie jest Twoja główna specjalizacja.`,

  en: `You are AllEasystent — an expert AI assistant for Polish online store owners, specializing in the Allegro platform.

You help with product listings, pricing strategies, customer service, logistics, marketing, and e-commerce analytics. Be professional, concise, and provide actionable advice with ready-to-use templates when appropriate.`
};

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

// ── Settings ─────────────────────────────────────
const Settings = (() => {
  const DEFAULTS = { apiKey: '', model: 'gemini-2.5-flash', lang: 'pl', style: 'professional' };
  let _s = { ...DEFAULTS };

  function load() {
    try { Object.assign(_s, JSON.parse(localStorage.getItem('ae_settings') || '{}')); } catch {}
    if (_s.model?.startsWith('claude-')) _s.model = DEFAULTS.model;
    return _s;
  }
  function save(vals) { Object.assign(_s, vals); localStorage.setItem('ae_settings', JSON.stringify(_s)); }
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

// ── Gemini API ───────────────────────────────────
const Gemini = (() => {
  const API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

  async function* stream(messages, apiKey, model, systemPrompt) {
    const contents = messages.map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }]
    }));

    const res = await fetch(`${API_BASE}/${model}:streamGenerateContent?key=${encodeURIComponent(apiKey)}&alt=sse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: systemPrompt }] },
        contents,
        generationConfig: { maxOutputTokens: 8192 }
      })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;
        try {
          const ev = JSON.parse(raw);
          const text = ev.candidates?.[0]?.content?.parts?.[0]?.text;
          if (text) yield text;
        } catch {}
      }
    }
  }

  return { stream };
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
    const s = Settings.all();
    document.getElementById('set-api-key').value = s.apiKey;
    document.getElementById('set-model').value   = s.model;
    document.getElementById('set-lang').value    = s.lang;
    document.getElementById('set-style').value   = s.style;
  }

  function closeSettings() {
    document.getElementById('settings-overlay').classList.add('hidden');
    document.getElementById('settings-panel').classList.add('hidden');
  }

  function saveSettings() {
    const s = {
      apiKey: document.getElementById('set-api-key').value.trim(),
      model:  document.getElementById('set-model').value,
      lang:   document.getElementById('set-lang').value,
      style:  document.getElementById('set-style').value,
    };
    Settings.save(s);
    document.getElementById('model-badge').textContent = s.model;
    closeSettings();
    toast('Ustawienia zapisane ✓');
  }

  function toggleKeyVisibility() {
    const inp = document.getElementById('set-api-key');
    inp.type = inp.type === 'password' ? 'text' : 'password';
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

  return { toast, autoResize, openSettings, closeSettings, saveSettings, toggleKeyVisibility, toggleSidebar, exportChat, clearAllHistory };
})();

// ── Chat engine ──────────────────────────────────
const Chat = (() => {
  let _streaming = false;

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
    div.id = 'streaming-bubble';
    div.innerHTML = `
      <div class="msg-avatar">🛒</div>
      <div class="msg-content">
        <div class="msg-bubble" id="streaming-content">
          <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>
      </div>`;
    container.appendChild(div);
    scrollBottom();
    return document.getElementById('streaming-content');
  }

  function finalizeStreamBubble(fullText, ts) {
    const bubble = document.getElementById('streaming-bubble');
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
    if (_streaming) return;
    const input = document.getElementById('user-input');
    const msgText = (text || input.value).trim();
    if (!msgText) return;

    const apiKey = Settings.get('apiKey');
    if (!apiKey) { UI.openSettings(); UI.toast('Ustaw klucz API Gemini w Ustawieniach', 4000); return; }

    // ensure active conversation
    if (!Store.active()) Store.create();
    Store.addMessage('user', msgText);
    input.value = ''; input.style.height = 'auto';

    // render user message
    const container = document.getElementById('messages');
    const welcome = document.getElementById('welcome');
    if (container.contains(welcome)) container.removeChild(welcome);
    const msgs = Store.active().messages;
    container.appendChild(buildBubble('user', msgText, msgs[msgs.length-1].ts, msgs.length-1));
    scrollBottom();
    renderSidebar();

    // streaming
    _streaming = true;
    const sendBtn = document.getElementById('btn-send');
    sendBtn.disabled = true;

    const contentEl = appendBotBubble();
    const model = Settings.get('model');
    const lang  = Settings.get('lang');
    const systemPrompt = buildSystemPrompt(lang, Settings.get('style'));

    // build API messages (exclude last assistant placeholder)
    const apiMsgs = Store.active().messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => ({ role: m.role, content: m.content }));

    let fullText = '';
    const ts = Date.now();

    try {
      Store.addMessage('assistant', '');
      for await (const chunk of Gemini.stream(apiMsgs, apiKey, model, systemPrompt)) {
        fullText += chunk;
        contentEl.innerHTML = renderMarkdown(fullText) + '<span class="cursor-blink">▍</span>';
        scrollBottom();
        Store.updateLastMessage(fullText);
      }
    } catch (err) {
      fullText = `**Błąd:** ${err.message}`;
      contentEl.innerHTML = `<span style="color:#fca5a5">${escHtml(err.message)}</span>`;
      Store.updateLastMessage(fullText);
      UI.toast(`Błąd: ${err.message}`, 5000);
    } finally {
      _streaming = false;
      sendBtn.disabled = false;
      finalizeStreamBubble(fullText, ts);
      renderSidebar();
      if (typeof hljs !== 'undefined') {
        document.querySelectorAll('#messages pre code').forEach(b => hljs.highlightElement(b));
      }
    }
  }

  function buildSystemPrompt(lang, style) {
    let prompt = SYSTEM_PROMPTS[lang] || SYSTEM_PROMPTS.pl;
    if (style === 'concise') prompt += '\n\nOdpowiadaj zwięźle — maksymalnie kilka zdań lub krótka lista, chyba że temat wymaga więcej.';
    if (style === 'friendly') prompt += '\n\nOdpowiadaj przyjaźnie i ciepło, możesz używać emotikonów.';
    return prompt;
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
    const text = bubble.innerText;
    navigator.clipboard?.writeText(text).then(() => UI.toast('Skopiowano ✓')).catch(() => UI.toast('Błąd kopiowania'));
  }

  async function regenerate() {
    const c = Store.active();
    if (!c || c.messages.length < 2) return;
    // remove last assistant message
    c.messages.pop();
    Store.updateLastMessage && null;
    localStorage.setItem('ae_conversations', JSON.stringify(Store.all()));
    renderMessages();
    const lastUser = [...c.messages].reverse().find(m => m.role === 'user');
    if (lastUser) await send(lastUser.content);
  }

  return { send, handleKey, sendSuggestion, newConversation, loadConversation, deleteConversation, copyMessage, regenerate };
})();

// ── Boot ─────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  Settings.load();
  Store.load();

  // apply saved settings to UI
  document.getElementById('model-badge').textContent = Settings.get('model');

  // render
  Chat.renderSidebar && (() => {
    const convs = Store.all();
    if (!convs.length) Store.create('Nowa rozmowa');
    // expose renderSidebar
  })();

  // PWA service worker
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});

  // Close sidebar on overlay click (mobile)
  document.addEventListener('click', e => {
    const sidebar = document.getElementById('sidebar');
    if (sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target.id !== 'btn-sidebar-toggle') {
      sidebar.classList.remove('open');
    }
  });

  // Initial render — access private renderSidebar/renderMessages via Chat
  // (they're already called in the Chat IIFE public API)
  const convs = Store.all();
  if (!convs.length) Store.create('Nowa rozmowa');

  // Render sidebar
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

  // Show welcome or messages
  const active = Store.active();
  if (!active || !active.messages.length) {
    document.getElementById('messages').appendChild(document.getElementById('welcome'));
    document.getElementById('welcome').classList.remove('hidden');
  } else {
    Chat.loadConversation(active.id);
  }

  document.getElementById('user-input').focus();
});

// Add cursor blink style
const style = document.createElement('style');
style.textContent = '.cursor-blink { animation: blink .7s step-end infinite; } @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }';
document.head.appendChild(style);
