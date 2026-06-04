const SYSTEM_PROMPTS={pl:`Jesteś AllEasystent — ekspert AI dla polskich właścicieli sklepów internetowych, specjalizujący się w platformie Allegro.

Pomagasz w:
- **Oferty i produkty**: tytuły SEO, opisy, parametry, zdjęcia, kategorie
- **Sprzedaż**: strategie cenowe, promocje, Allegro Ads, analiza konkurencji
- **Obsługa klienta**: szablony wiadomości, odpowiedzi na pytania, reklamacje i zwroty
- **Logistyka**: metody wysyłki, Allegro Smart, pakowanie, zarządzanie magazynem
- **Marketing**: kampanie, social media, cross-selling, upselling
- **Analityka**: interpretacja statystyk, prognozowanie, optymalizacja konwersji
- **Prawo e-commerce**: regulaminy, RODO, prawa konsumenta

Odpowiadasz po polsku, profesjonalnie i konkretnie. Używasz formatowania markdown — nagłówków, list i pogrubień — gdy poprawia to czytelność. Podajesz gotowe do użycia szablony i przykłady.`,en:`You are AllEasystent — an expert AI assistant for Polish online store owners, specializing in the Allegro platform. Be professional, concise, and provide actionable advice with ready-to-use templates when appropriate.`};

if(typeof marked!=='undefined'){marked.setOptions({breaks:true,gfm:true});}

const Settings=(()=>{
  const DEFAULTS={apiKey:'',model:'claude-sonnet-4-6',lang:'pl',style:'professional'};
  let _s={...DEFAULTS};
  function load(){try{Object.assign(_s,JSON.parse(localStorage.getItem('ae_settings')||'{}'))}catch{}return _s;}
  function save(v){Object.assign(_s,v);localStorage.setItem('ae_settings',JSON.stringify(_s));}
  function get(k){return _s[k];}
  return{load,save,get,all:()=>({..._s})};
})();

const Store=(()=>{
  const KEY='ae_conversations';
  let convs=[],activeId=null;
  function load(){try{convs=JSON.parse(localStorage.getItem(KEY)||'[]')}catch{convs=[];}if(convs.length)activeId=convs[0].id;}
  function save(){localStorage.setItem(KEY,JSON.stringify(convs));}
  function create(title='Nowa rozmowa'){const c={id:Date.now().toString(),title,messages:[],createdAt:Date.now()};convs.unshift(c);activeId=c.id;save();return c;}
  function active(){return convs.find(c=>c.id===activeId)||null;}
  function setActive(id){activeId=id;return active();}
  function addMessage(role,content){const c=active();if(!c)return;c.messages.push({role,content,ts:Date.now()});if(c.messages.length===2&&role==='assistant')c.title=c.messages[0].content.slice(0,50).replace(/\n/g,' ');save();}
  function updateLast(content){const c=active();if(!c||!c.messages.length)return;const l=c.messages[c.messages.length-1];if(l.role==='assistant')l.content=content;save();}
  function del(id){convs=convs.filter(c=>c.id!==id);if(activeId===id)activeId=convs[0]?.id||null;save();}
  function clearAll(){convs=[];activeId=null;localStorage.removeItem(KEY);}
  return{load,create,active,setActive,addMessage,updateLast,del,clearAll,all:()=>convs};
})();

const Claude=(()=>{
  const API='https://api.anthropic.com/v1/messages';
  async function* stream(messages,apiKey,model,sys){
    const res=await fetch(API,{method:'POST',headers:{'Content-Type':'application/json','x-api-key':apiKey,'anthropic-version':'2023-06-01','anthropic-dangerous-allow-browser':'true'},body:JSON.stringify({model,max_tokens:4096,stream:true,system:sys,messages})});
    if(!res.ok){const e=await res.json().catch(()=>({}));throw new Error(e.error?.message||`HTTP ${res.status}`);}
    const reader=res.body.getReader(),dec=new TextDecoder();let buf='';
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        const raw=line.slice(6).trim();if(raw==='[DONE]')return;
        try{const ev=JSON.parse(raw);if(ev.type==='content_block_delta'&&ev.delta?.type==='text_delta')yield ev.delta.text;}catch{}
      }
    }
  }
  return{stream};
})();

const UI=(()=>{
  let _t=null;
  function toast(msg,ms=2500){clearTimeout(_t);const el=document.getElementById('toast');el.textContent=msg;el.classList.remove('hidden');_t=setTimeout(()=>el.classList.add('hidden'),ms);}
  function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,180)+'px';}
  function openSettings(){document.getElementById('settings-overlay').classList.remove('hidden');document.getElementById('settings-panel').classList.remove('hidden');const s=Settings.all();document.getElementById('set-api-key').value=s.apiKey;document.getElementById('set-model').value=s.model;document.getElementById('set-lang').value=s.lang;document.getElementById('set-style').value=s.style;}
  function closeSettings(){document.getElementById('settings-overlay').classList.add('hidden');document.getElementById('settings-panel').classList.add('hidden');}
  function saveSettings(){const s={apiKey:document.getElementById('set-api-key').value.trim(),model:document.getElementById('set-model').value,lang:document.getElementById('set-lang').value,style:document.getElementById('set-style').value};Settings.save(s);document.getElementById('model-badge').textContent=s.model;closeSettings();toast('Ustawienia zapisane ✓');}
  function toggleKeyVisibility(){const i=document.getElementById('set-api-key');i.type=i.type==='password'?'text':'password';}
  function toggleSidebar(){document.getElementById('sidebar').classList.toggle('open');}
  function exportChat(){const c=Store.active();if(!c||!c.messages.length){toast('Brak wiadomości do eksportu');return;}const text=c.messages.map(m=>`[${m.role==='user'?'Ty':'AllEasystent'}]\n${m.content}`).join('\n\n---\n\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/plain'}));a.download=`alleasystent-${new Date().toISOString().slice(0,10)}.txt`;a.click();}
  function clearAllHistory(){if(!confirm('Usunąć całą historię rozmów?'))return;Store.clearAll();Chat.newConversation();closeSettings();toast('Historia usunięta');}
  return{toast,autoResize,openSettings,closeSettings,saveSettings,toggleKeyVisibility,toggleSidebar,exportChat,clearAllHistory};
})();

const Chat=(()=>{
  let _streaming=false;

  function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
  function md(text){return typeof marked!=='undefined'?marked.parse(text):esc(text).replace(/\n/g,'<br>');}
  function scrollBottom(){const el=document.getElementById('messages');el.scrollTop=el.scrollHeight;}

  function renderSidebar(){const list=document.getElementById('sidebar-history');const all=Store.all();const active=Store.active();list.innerHTML=all.length?all.map(c=>`<div class="history-item ${c.id===active?.id?'active':''}" onclick="Chat.loadConversation('${c.id}')"><span class="hi-icon">💬</span><span style="overflow:hidden;text-overflow:ellipsis;flex:1">${esc(c.title)}</span><button class="hi-del" onclick="event.stopPropagation();Chat.deleteConversation('${c.id}')" title="Usuń">✕</button></div>`).join(''):'<p style="color:var(--muted);font-size:.8rem;padding:.5rem .75rem">Brak rozmów</p>';}

  function buildBubble(role,content,ts,idx){
    const isUser=role==='user';
    const div=document.createElement('div');
    div.className=`msg msg-${isUser?'user':'bot'}`;
    const html=isUser?esc(content).replace(/\n/g,'<br>'):md(content);
    const time=ts?new Date(ts).toLocaleTimeString('pl-PL',{hour:'2-digit',minute:'2-digit'}):'';
    div.innerHTML=`<div class="msg-avatar">${isUser?'👤':'🛒'}</div><div class="msg-content"><div class="msg-bubble">${html}</div><div class="msg-actions"><button class="msg-act-btn" onclick="Chat.copyMessage(this)">📋 Kopiuj</button>${!isUser?'<button class="msg-act-btn" onclick="Chat.regenerate()">↺ Nowa odpowiedź</button>':''}</div>${time?`<span class="msg-time">${time}</span>`:''}</div>`;
    return div;
  }

  function renderMessages(){
    const c=Store.active();
    const container=document.getElementById('messages');
    const welcome=document.getElementById('welcome');
    container.innerHTML='';
    if(!c||!c.messages.length){container.appendChild(welcome);return;}
    c.messages.forEach((m,i)=>container.appendChild(buildBubble(m.role,m.content,m.ts,i)));
    if(typeof hljs!=='undefined')container.querySelectorAll('pre code').forEach(b=>hljs.highlightElement(b));
    scrollBottom();
  }

  function appendBot(){
    const container=document.getElementById('messages');
    const welcome=document.getElementById('welcome');
    if(container.contains(welcome))container.removeChild(welcome);
    const div=document.createElement('div');
    div.className='msg msg-bot';div.id='streaming-bubble';
    div.innerHTML='<div class="msg-avatar">🛒</div><div class="msg-content"><div class="msg-bubble" id="sc"><div class="typing-dots"><span></span><span></span><span></span></div></div></div>';
    container.appendChild(div);scrollBottom();
    return document.getElementById('sc');
  }

  function finalizeStream(text,ts){
    const b=document.getElementById('streaming-bubble');if(!b)return;
    const idx=Store.active()?.messages.length-1;
    const r=buildBubble('assistant',text,ts,idx);
    b.replaceWith(r);
    if(typeof hljs!=='undefined')r.querySelectorAll('pre code').forEach(b=>hljs.highlightElement(b));
  }

  function buildSys(lang,style){
    let p=SYSTEM_PROMPTS[lang]||SYSTEM_PROMPTS.pl;
    if(style==='concise')p+='\n\nOdpowiadaj zwieźle — maksymalnie kilka zdań lub krótka lista.';
    if(style==='friendly')p+='\n\nOdpowiadaj przyjaźnie i ciepło, możesz używać emotikon.';
    return p;
  }

  async function send(text){
    if(_streaming)return;
    const input=document.getElementById('user-input');
    const msg=(text||input.value).trim();if(!msg)return;
    const apiKey=Settings.get('apiKey');
    if(!apiKey){UI.openSettings();UI.toast('Ustaw klucz API Claude w Ustawieniach',4000);return;}
    if(!Store.active())Store.create();
    Store.addMessage('user',msg);
    input.value='';input.style.height='auto';
    const container=document.getElementById('messages');
    const welcome=document.getElementById('welcome');
    if(container.contains(welcome))container.removeChild(welcome);
    const msgs=Store.active().messages;
    container.appendChild(buildBubble('user',msg,msgs[msgs.length-1].ts,msgs.length-1));
    scrollBottom();renderSidebar();
    _streaming=true;
    document.getElementById('btn-send').disabled=true;
    const contentEl=appendBot();
    const model=Settings.get('model'),lang=Settings.get('lang'),sys=buildSys(lang,Settings.get('style'));
    const apiMsgs=Store.active().messages.filter(m=>m.role==='user'||m.role==='assistant').map(m=>({role:m.role,content:m.content}));
    let full='';const ts=Date.now();
    try{
      Store.addMessage('assistant','');
      for await(const chunk of Claude.stream(apiMsgs,apiKey,model,sys)){
        full+=chunk;
        contentEl.innerHTML=md(full)+'<span style="animation:blink .7s step-end infinite">◍</span>';
        scrollBottom();Store.updateLast(full);
      }
    }catch(err){
      full=`**Błąd:** ${err.message}`;
      contentEl.innerHTML=`<span style="color:#fca5a5">${esc(err.message)}</span>`;
      Store.updateLast(full);UI.toast(`Błąd: ${err.message}`,5000);
    }finally{
      _streaming=false;
      document.getElementById('btn-send').disabled=false;
      finalizeStream(full,ts);renderSidebar();
    }
  }

  function handleKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}
  function sendSuggestion(btn){send(btn.textContent);}

  function newConversation(){Store.create();renderSidebar();renderMessages();document.getElementById('sidebar').classList.remove('open');document.getElementById('user-input').focus();}
  function loadConversation(id){Store.setActive(id);renderSidebar();renderMessages();document.getElementById('sidebar').classList.remove('open');}
  function deleteConversation(id){Store.del(id);renderSidebar();renderMessages();}

  function copyMessage(btn){const bubble=btn.closest('.msg-content').querySelector('.msg-bubble');navigator.clipboard?.writeText(bubble.innerText).then(()=>UI.toast('Skopiowano ✓')).catch(()=>UI.toast('Błąd kopiowania'));}

  async function regenerate(){
    const c=Store.active();if(!c||c.messages.length<2)return;
    c.messages.pop();localStorage.setItem('ae_conversations',JSON.stringify(Store.all()));
    renderMessages();
    const last=[...c.messages].reverse().find(m=>m.role==='user');
    if(last)await send(last.content);
  }

  return{send,handleKey,sendSuggestion,newConversation,loadConversation,deleteConversation,copyMessage,regenerate,renderSidebar,renderMessages};
})();

const _css=document.createElement('style');
_css.textContent='@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}';
document.head.appendChild(_css);

window.addEventListener('DOMContentLoaded',()=>{
  Settings.load();Store.load();
  document.getElementById('model-badge').textContent=Settings.get('model');
  if(!Store.all().length)Store.create('Nowa rozmowa');
  Chat.renderSidebar();
  const active=Store.active();
  if(!active||!active.messages.length){document.getElementById('messages').appendChild(document.getElementById('welcome'));}
  else Chat.loadConversation(active.id);
  if('serviceWorker' in navigator)navigator.serviceWorker.register('sw.js').catch(()=>{});
  document.addEventListener('click',e=>{const sb=document.getElementById('sidebar');if(sb.classList.contains('open')&&!sb.contains(e.target)&&e.target.id!=='btn-sidebar-toggle')sb.classList.remove('open');});
  document.getElementById('user-input').focus();
});