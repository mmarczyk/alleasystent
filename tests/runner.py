"""
Test runner UI — serves a web interface for running pytest and streaming results.
"""

import asyncio
import os
import subprocess
import threading
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

app = FastAPI()

_lock = threading.Lock()
_running = False
_process: subprocess.Popen | None = None
_output: list[str] = []  # all lines since last run, replayed to new SSE clients


def _run_pytest():
    global _running, _process, _output
    with _lock:
        _running = True
        _output = []
        _process = None

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        ["pytest", "-v", "--tb=short", "--no-header", "-p", "no:color"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd="/tests",
        env=env,
    )
    with _lock:
        _process = proc

    for raw in proc.stdout:
        line = raw.rstrip()
        with _lock:
            _output.append(line)

    proc.wait()
    with _lock:
        _output.append(f"__EXIT__{proc.returncode}")
        _running = False
        _process = None


@app.post("/run")
async def run():
    with _lock:
        if _running:
            return JSONResponse({"status": "already_running"})
    threading.Thread(target=_run_pytest, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.post("/stop")
async def stop():
    with _lock:
        proc = _process
    if proc:
        proc.terminate()
        return JSONResponse({"status": "stopped"})
    return JSONResponse({"status": "not_running"})


@app.get("/stream")
async def stream():
    async def generate():
        pos = 0
        while True:
            with _lock:
                chunk = _output[pos:]
                is_running = _running

            for line in chunk:
                yield f"data: {line}\n\n"
                pos += 1
                if line.startswith("__EXIT__"):
                    return

            if not is_running and pos >= len(_output) and _output:
                return

            await asyncio.sleep(0.2)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/status")
async def status():
    with _lock:
        return {"running": _running, "lines": len(_output)}


HTML = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AllEasystent — Test Runner</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }

  header { padding: 20px 32px; border-bottom: 1px solid #1e2535; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.2rem; font-weight: 600; color: #fff; }
  header span { font-size: 0.8rem; color: #64748b; }

  .controls { padding: 20px 32px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  button { padding: 8px 20px; border: none; border-radius: 6px; font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: opacity .15s; }
  button:hover { opacity: 0.85; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  #btn-run  { background: #22c55e; color: #fff; }
  #btn-stop { background: #ef4444; color: #fff; }
  #status-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; background: #1e2535; color: #94a3b8; }
  #status-badge.running { background: #1e3a5f; color: #60a5fa; }
  #status-badge.passed  { background: #14532d; color: #4ade80; }
  #status-badge.failed  { background: #450a0a; color: #f87171; }

  .summary-bar { padding: 0 32px 20px; display: flex; gap: 16px; flex-wrap: wrap; }
  .stat { padding: 8px 16px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; background: #1e2535; }
  .stat.pass { background: #14532d; color: #4ade80; }
  .stat.fail { background: #450a0a; color: #f87171; }
  .stat.skip { background: #1c1917; color: #a8a29e; }

  .main { padding: 0 32px 32px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 900px) { .main { grid-template-columns: 1fr; } }

  .panel { background: #1a1f2e; border: 1px solid #1e2535; border-radius: 10px; overflow: hidden; }
  .panel-header { padding: 12px 16px; font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid #1e2535; }

  #test-list { list-style: none; max-height: 500px; overflow-y: auto; }
  #test-list li { padding: 9px 16px; font-size: 0.82rem; font-family: monospace; border-bottom: 1px solid #0f1117; display: flex; align-items: baseline; gap: 8px; line-height: 1.4; }
  #test-list li:last-child { border-bottom: none; }
  .icon { font-size: 1rem; flex-shrink: 0; }
  .test-name { color: #94a3b8; word-break: break-all; }
  .test-name .module { color: #475569; }
  li.pass .test-name { color: #e2e8f0; }
  li.fail .test-name { color: #fca5a5; }
  li.skip .test-name { color: #78716c; }
  li.running .test-name { color: #93c5fd; }

  #log { font-family: monospace; font-size: 0.78rem; line-height: 1.6; padding: 12px 16px; max-height: 500px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; color: #64748b; }
  .log-pass { color: #4ade80; }
  .log-fail { color: #f87171; }
  .log-skip { color: #a8a29e; }
  .log-sep  { color: #334155; }
  .log-error { color: #fbbf24; }
</style>
</head>
<body>

<header>
  <h1>AllEasystent — Test Runner</h1>
  <span id="target-url"></span>
</header>

<div class="controls">
  <button id="btn-run" onclick="startRun()">▶ Uruchom testy</button>
  <button id="btn-stop" onclick="stopRun()" disabled>■ Stop</button>
  <span id="status-badge">Gotowy</span>
</div>

<div class="summary-bar" id="summary-bar" style="display:none">
  <span class="stat pass" id="stat-pass">✓ 0 passed</span>
  <span class="stat fail" id="stat-fail">✗ 0 failed</span>
  <span class="stat skip" id="stat-skip">– 0 skipped</span>
  <span class="stat" id="stat-time"></span>
</div>

<div class="main">
  <div class="panel">
    <div class="panel-header">Wyniki testów</div>
    <ul id="test-list"><li style="color:#475569;padding:16px">Brak wyników — uruchom testy.</li></ul>
  </div>
  <div class="panel">
    <div class="panel-header">Logi</div>
    <div id="log">Tutaj pojawi się output pytest...</div>
  </div>
</div>

<script>
const targetUrl = (window.ENV_ALLEASYSTENT_URL || '') || '(nie ustawiono ALLEASYSTENT_URL)';
document.getElementById('target-url').textContent = '→ ' + targetUrl;

let es = null;
let stats = {pass:0, fail:0, skip:0};
let currentRun = [];

function setBadge(state, text) {
  const b = document.getElementById('status-badge');
  b.className = state;
  b.textContent = text;
}

function setButtons(running) {
  document.getElementById('btn-run').disabled = running;
  document.getElementById('btn-stop').disabled = !running;
}

function clearResults() {
  document.getElementById('test-list').innerHTML = '';
  document.getElementById('log').innerHTML = '';
  document.getElementById('summary-bar').style.display = 'none';
  stats = {pass:0, fail:0, skip:0};
  currentRun = [];
}

function addTestResult(name, state) {
  const icons = {pass:'✅', fail:'❌', skip:'⏭️', running:'⏳', error:'💥'};
  const li = document.createElement('li');
  li.className = state;
  const parts = name.split('::');
  const shortName = parts.slice(1).join(' › ') || name;
  const module = parts[0] ? parts[0].replace('.py','') + ' › ' : '';
  li.innerHTML = `<span class="icon">${icons[state]||'•'}</span><span class="test-name"><span class="module">${module}</span>${shortName}</span>`;
  document.getElementById('test-list').appendChild(li);
  li.scrollIntoView({block:'nearest'});
  return li;
}

function appendLog(line) {
  const log = document.getElementById('log');
  const div = document.createElement('div');
  let cls = '';
  if (/PASSED/.test(line)) cls = 'log-pass';
  else if (/FAILED|ERROR/.test(line)) cls = 'log-fail';
  else if (/SKIPPED/.test(line)) cls = 'log-skip';
  else if (/^=+/.test(line)) cls = 'log-sep';
  else if (/^E /.test(line)) cls = 'log-error';
  if (cls) div.className = cls;
  div.textContent = line;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function updateStats() {
  document.getElementById('stat-pass').textContent = `✓ ${stats.pass} passed`;
  document.getElementById('stat-fail').textContent = `✗ ${stats.fail} failed`;
  document.getElementById('stat-skip').textContent = `– ${stats.skip} skipped`;
  document.getElementById('summary-bar').style.display = 'flex';
}

function parseLine(line) {
  // Test result line: "test_01_routing.py::Class::method PASSED" etc.
  const m = line.match(/^(\\S+\\.py(?:::\\S+)+)\\s+(PASSED|FAILED|ERROR|SKIPPED)/);
  if (m) {
    const state = {PASSED:'pass',FAILED:'fail',ERROR:'error',SKIPPED:'skip'}[m[2]];
    addTestResult(m[1], state);
    if (state === 'pass') stats.pass++;
    else if (state === 'fail' || state === 'error') stats.fail++;
    else if (state === 'skip') stats.skip++;
    updateStats();
    return;
  }
  // Summary line
  const s = line.match(/(\\d+) passed|( \\d+) failed|(\\d+) skipped|in ([\\d.]+)s/g);
  if (s && /=====/.test(line)) {
    const t = line.match(/in ([\\d.]+)s/);
    if (t) document.getElementById('stat-time').textContent = `⏱ ${t[1]}s`;
  }
  appendLog(line);
}

async function startRun() {
  clearResults();
  setButtons(true);
  setBadge('running', '⏳ Trwa...');

  const r = await fetch('/run', {method:'POST'});
  const j = await r.json();
  if (j.status === 'already_running') { setButtons(true); return; }

  if (es) es.close();
  es = new EventSource('/stream');
  es.onmessage = (e) => {
    const line = e.data;
    if (line === '__PING__') return;
    if (line.startsWith('__EXIT__')) {
      const code = parseInt(line.replace('__EXIT__',''));
      es.close(); es = null;
      setButtons(false);
      if (code === 0) setBadge('passed', '✅ Wszystkie testy przeszły');
      else setBadge('failed', `❌ Testy nie przeszły (kod ${code})`);
      return;
    }
    parseLine(line);
  };
  es.onerror = () => { setButtons(false); setBadge('', 'Błąd połączenia'); };
}

async function stopRun() {
  await fetch('/stop', {method:'POST'});
  if (es) { es.close(); es = null; }
  setButtons(false);
  setBadge('', 'Zatrzymano');
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML
