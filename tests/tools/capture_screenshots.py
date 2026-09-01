from __future__ import annotations

"""Replay every tool result through the real PWA and screenshot what it renders.

    python -m tests.tools.capture_screenshots --out artifacts/tool-screenshots

What it does:
  1. runs the tool matrix (tests/tools/runner.py) against the fake Allegro API,
  2. serves web/ from a local stub backend whose POST /query hands back the tool
     output for the case being captured (agent = "allegro:<output_format>", the
     same string the orchestrator sends in production),
  3. drives Chromium through the real chat UI — types the seller's question,
     waits for the reply — and screenshots the chat bubble and, when the reply
     opens one, the full-screen document viewer,
  4. writes index.html: a gallery of every case with its screenshots, arguments,
     API calls and raw tool output.

Caveat worth remembering when reading the screenshots: this pipeline stops at
the tool boundary. In production the tool's text is handed to Gemini, which
rewrites it into the final answer (a markdown table for "table" format, a
dashboard for "dashboard", …) before it reaches the browser. No API key is
needed here and no model runs, so what you see is the RAW tool payload rendered
by the real UI — the data and the UI chrome are production-accurate, the
model's prose/table styling on top of it is not.
"""

import argparse
import asyncio
import base64
import http.server
import io
import json
import socketserver
import threading
import time
from datetime import datetime
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"

import sys  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.tools.runner import run_all  # noqa: E402

# The sandbox ships a Chromium build that does not match the pip-installed
# Playwright's expected revision, so the binary is passed explicitly when it is
# there; elsewhere Playwright's own download is used.
CHROMIUM_PATH = Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome")

VIEWPORT_W = 1024
MIN_H = 640
MAX_H = 3600


# ── Stub backend ─────────────────────────────────────────────────────────────


class _State:
    """The reply the stub backend serves for the case currently being captured."""

    text = ""
    fmt = "chat"


def _make_handler(state: _State):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        def log_message(self, *args):  # silence per-request logging
            pass

        def _send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                return self._send_json({"status": "ok", "git_sha": "tool-matrix"})
            if self.path.startswith("/auth/me"):
                return self._send_json({"detail": "unauthorized"}, 401)
            if self.path == "/sw.js":
                return self._send_json({"detail": "disabled in capture"}, 404)
            return super().do_GET()

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if self.path == "/__case":
                payload = json.loads(raw)
                state.text = payload["text"]
                state.fmt = payload["format"]
                return self._send_json({"ok": True})
            if self.path.startswith("/query"):
                return self._send_json({
                    "response": state.text,
                    "agent": f"allegro:{state.fmt}",
                    "session_id": "capture",
                })
            return self._send_json({"ok": True})

    return Handler


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_stub_server(state: _State) -> tuple[_Server, int]:
    server = _Server(("127.0.0.1", 0), _make_handler(state))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


# ── Browser capture ──────────────────────────────────────────────────────────


def _fake_jwt() -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    payload = {
        "sub": "demo_seller",
        "name": "ElektroDom (demo)",
        "exp": int(time.time()) + 86400,
    }
    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(payload)}.signature"


_MONITOR_LS_KEYS = {
    "order": "ae_monitor_enabled",
    "message": "ae_message_monitor_enabled",
    "returns": "ae_returns_monitor_enabled",
    "invoice_reminder": "ae_invoice_reminder_enabled",
}


def capture(results: list[dict], out_dir: Path) -> list[dict]:
    from playwright.sync_api import sync_playwright

    shots_dir = out_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    state = _State()
    server, port = start_stub_server(state)
    base_url = f"http://127.0.0.1:{port}/index.html"

    try:
        with sync_playwright() as pw:
            launch_kwargs = {"args": ["--force-color-profile=srgb"]}
            if CHROMIUM_PATH.exists():
                launch_kwargs["executable_path"] = str(CHROMIUM_PATH)
            browser = pw.chromium.launch(**launch_kwargs)

            for result in results:
                # Fresh context per case: no leftover conversation history,
                # monitoring flags or doc-viewer tabs from the previous one.
                context = browser.new_context(
                    viewport={"width": VIEWPORT_W, "height": MIN_H},
                    device_scale_factor=2,
                    locale="pl-PL",
                    timezone_id="Europe/Warsaw",
                )
                # The service worker would cache the stub responses between cases.
                context.route("**/sw.js", lambda route: route.abort())
                monitors = {**{k: False for k in _MONITOR_LS_KEYS}, **(result["monitors"] or {})}
                token = _fake_jwt()
                context.add_init_script(f"""
                  try {{
                    localStorage.clear();
                    localStorage.setItem('ae_session_token', {json.dumps(token)});
                    const flags = {json.dumps({_MONITOR_LS_KEYS[k]: v for k, v in monitors.items()})};
                    for (const [k, v] of Object.entries(flags)) {{
                      if (v) localStorage.setItem(k, '1');
                    }}
                  }} catch (e) {{}}
                """)
                page = context.new_page()

                # Tell the stub backend which reply to serve, then ask the question.
                page.request.post(f"http://127.0.0.1:{port}/__case", data=json.dumps(
                    {"text": result["output"], "format": result["format"]}
                ))
                page.set_viewport_size({"width": VIEWPORT_W, "height": MIN_H})
                page.goto(base_url, wait_until="networkidle")
                page.wait_for_selector("#user-input", state="visible", timeout=15000)

                page.fill("#user-input", result["question"])
                page.click("#btn-send")
                page.wait_for_selector("#waiting-bubble", state="detached", timeout=20000)
                page.wait_for_timeout(350)

                doc_open = page.evaluate(
                    "!document.getElementById('doc-viewer').classList.contains('hidden')"
                )

                # Chat view — with the document viewer (if any) closed.
                if doc_open:
                    page.evaluate("DocViewer.close()")
                    page.wait_for_timeout(150)
                chat_shot = shots_dir / f"{result['id']}--chat.png"
                _shot_fitting(page, "#messages", chat_shot)
                result["shot_chat"] = chat_shot.relative_to(out_dir).as_posix()

                # Document viewer — the "artifact" presentation for table /
                # document / dashboard replies.
                if doc_open:
                    page.evaluate(
                        """() => {
                            const el = document.querySelector('.msg-bubble[data-doc-key]');
                            if (el) DocViewer.openFromKey(Number(el.dataset.docKey));
                        }"""
                    )
                    page.wait_for_timeout(300)
                    doc_shot = shots_dir / f"{result['id']}--doc.png"
                    _shot_fitting(page, ".doc-body", doc_shot)
                    result["shot_doc"] = doc_shot.relative_to(out_dir).as_posix()
                else:
                    result["shot_doc"] = None
                result["doc_viewer_opened"] = doc_open
                print(f"  📸 {result['id']}" + ("  (+ doc viewer)" if doc_open else ""))
                context.close()

            browser.close()
    finally:
        server.shutdown()
    return results


def _shot_fitting(page, selector: str, path: Path) -> None:
    """Grow the viewport until the scrollable pane fits, then screenshot it.

    The app is a fixed-height flex layout with its own inner scrolling, so a
    full-page screenshot would only ever show the first screenful.
    """
    needed = page.evaluate(
        """(sel) => {
            const el = document.querySelector(sel);
            if (!el) return 0;
            const extra = window.innerHeight - el.clientHeight;
            return Math.ceil(el.scrollHeight + extra + 8);
        }""",
        selector,
    )
    height = max(MIN_H, min(int(needed or MIN_H), MAX_H))
    page.set_viewport_size({"width": VIEWPORT_W, "height": height})
    page.wait_for_timeout(250)
    page.evaluate("(sel) => { const el = document.querySelector(sel); if (el) el.scrollTop = 0; }", selector)
    page.screenshot(path=str(path))
    page.set_viewport_size({"width": VIEWPORT_W, "height": MIN_H})


# ── Gallery ──────────────────────────────────────────────────────────────────

_FORMAT_LABEL = {
    "chat": "czat — odpowiedź w dymku",
    "table": "tabela — model formatuje dane w tabelę markdown",
    "dashboard": "dashboard — sekcje + wykresy",
    "document": "dokument — pełny widok",
    "action": "akcja — komunikat + przycisk w UI",
}


def _image_src(path: Path, embed: bool, rel: str) -> str:
    """Either a relative <img src>, or a self-contained WebP data URI.

    WebP at quality 78 keeps the UI text crisp while making a 55-screenshot
    gallery small enough to hand over as a single file.
    """
    if not embed:
        return rel
    try:
        from PIL import Image

        buf = io.BytesIO()
        with Image.open(path) as im:
            im.convert("RGB").save(buf, format="WEBP", quality=78, method=5)
        return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001 — Pillow missing: fall back to the raw PNG
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build_gallery(results: list[dict], out_dir: Path, embed: bool = False) -> Path:
    rows = []
    for r in results:
        badge = "OK" if r["ok"] else "BŁĄD"
        badge_class = "ok" if r["ok"] else "fail"
        chat_src = _image_src(out_dir / r["shot_chat"], embed, r["shot_chat"])
        shots = (f'<figure><figcaption>Czat</figcaption>'
                 f'<img src="{chat_src}" alt="{r["id"]} — czat"></figure>')
        if r.get("shot_doc"):
            doc_src = _image_src(out_dir / r["shot_doc"], embed, r["shot_doc"])
            shots += (f'<figure><figcaption>Pełny widok (doc viewer)</figcaption>'
                      f'<img src="{doc_src}" alt="{r["id"]} — dokument"></figure>')
        api = "".join(f"<li><code>{escape(c)}</code></li>" for c in r["api_calls"]) or "<li>—</li>"
        rows.append(f"""
<section class="case" id="{r['id']}">
  <header>
    <h2>{escape(r['tool'])}<span class="case-id">{escape(r['id'])}</span></h2>
    <span class="badge {badge_class}">{badge}</span>
  </header>
  <p class="note">{escape(r['note'])}</p>
  <dl>
    <dt>Pytanie</dt><dd>„{escape(r['question'])}”</dd>
    <dt>Argumenty</dt><dd><code>{escape(json.dumps(r['args'], ensure_ascii=False))}</code></dd>
    <dt>Format odpowiedzi</dt><dd>{escape(_FORMAT_LABEL.get(r['format'], r['format']))}</dd>
    <dt>Czas / rozmiar</dt><dd>{r['duration_ms']} ms · {r['chars']} znaków</dd>
    <dt>Wywołania API</dt><dd><ul class="api">{api}</ul></dd>
  </dl>
  <div class="shots">{shots}</div>
  <details><summary>Surowa odpowiedź narzędzia</summary><pre>{escape(r['output'])}</pre></details>
</section>""")

    toc = "".join(
        f'<li><a href="#{r["id"]}">{escape(r["id"])}</a> '
        f'<span class="{"ok" if r["ok"] else "fail"}">{"✓" if r["ok"] else "✗"}</span></li>'
        for r in results
    )
    passed = sum(1 for r in results if r["ok"])
    generated = datetime.now().strftime("%d.%m.%Y %H:%M")

    html = f"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AlleAsystent — testy narzędzi</title>
<style>
  :root {{ color-scheme: dark; --bg:#0f0f1a; --card:#171728; --line:#2a2a40;
           --text:#e2e8f0; --muted:#94a3b8; --accent:#818cf8; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
         font:15px/1.6 'Segoe UI', system-ui, sans-serif; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:2rem 1.25rem 4rem; }}
  h1 {{ font-size:1.6rem; margin:0 0 .35rem; }}
  .lede {{ color:var(--muted); margin:0 0 1.5rem; }}
  .summary {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
              padding:1rem 1.25rem; margin-bottom:2rem; }}
  ul.toc {{ columns:3; list-style:none; padding:0; margin:.75rem 0 0; font-size:.85rem; }}
  ul.toc li {{ break-inside:avoid; }}
  ul.toc a {{ color:var(--accent); text-decoration:none; }}
  .ok {{ color:#34d399; }} .fail {{ color:#f87171; }}
  section.case {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
                  padding:1.25rem 1.25rem 1rem; margin-bottom:1.75rem; }}
  section.case header {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; }}
  section.case h2 {{ font-size:1.1rem; margin:0; font-family:'SFMono-Regular',Consolas,monospace; }}
  .case-id {{ display:block; font-size:.75rem; color:var(--muted); font-weight:400; }}
  .badge {{ font-size:.75rem; font-weight:700; padding:.2rem .55rem; border-radius:999px;
            border:1px solid currentColor; }}
  .note {{ color:var(--muted); margin:.5rem 0 1rem; }}
  dl {{ display:grid; grid-template-columns:max-content 1fr; gap:.3rem 1rem; margin:0 0 1rem;
        font-size:.88rem; }}
  dt {{ color:var(--muted); }} dd {{ margin:0; }}
  code {{ font-family:'SFMono-Regular',Consolas,monospace; font-size:.82rem; }}
  ul.api {{ margin:0; padding-left:1.1rem; }}
  .shots {{ display:flex; gap:1rem; flex-wrap:wrap; }}
  figure {{ margin:0; flex:1 1 380px; }}
  figcaption {{ font-size:.78rem; color:var(--muted); margin-bottom:.35rem; }}
  img {{ width:100%; border:1px solid var(--line); border-radius:10px; display:block; }}
  details {{ margin-top:1rem; }}
  summary {{ cursor:pointer; color:var(--accent); font-size:.85rem; }}
  pre {{ background:#0b0b14; border:1px solid var(--line); border-radius:8px; padding:1rem;
         overflow-x:auto; font-size:.8rem; white-space:pre-wrap; word-break:break-word; }}
</style></head><body><div class="wrap">
<h1>AlleAsystent — testy wszystkich narzędzi</h1>
<p class="lede">Każde narzędzie agenta Allegro uruchomione na syntetycznych danych sklepu
i wyrenderowane w prawdziwym interfejsie PWA. Wygenerowano {generated}.</p>
<div class="summary">
  <strong>{passed}/{len(results)}</strong> przypadków zakończonych powodzeniem ·
  {len({r['tool'] for r in results})} narzędzi.
  <p style="color:var(--muted);font-size:.85rem;margin:.75rem 0 0">
    Zrzuty pokazują <em>surowe dane zwracane przez narzędzie</em> wyrenderowane przez
    prawdziwy frontend. W produkcji ten tekst trafia jeszcze do Gemini, który układa
    z niego finalną odpowiedź (tabelę, dashboard, dokument) — ten krok jest tu pominięty,
    bo nie wymaga klucza API i nie jest deterministyczny.</p>
  <ul class="toc">{toc}</ul>
</div>
{''.join(rows)}
</div></body></html>"""

    path = out_dir / ("gallery-standalone.html" if embed else "index.html")
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Screenshot every Allegro tool's output")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "tool-screenshots")
    parser.add_argument("--results", type=Path, help="reuse a runner.py --json dump")
    parser.add_argument("--standalone", action="store_true",
                        help="also write gallery-standalone.html with the images inlined")
    args = parser.parse_args()

    if args.results and args.results.exists():
        results = json.loads(args.results.read_text())
    else:
        print("▶ uruchamiam matrycę narzędzi…")
        results = asyncio.run(run_all())

    failed = [r["id"] for r in results if not r["ok"]]
    print(f"▶ {len(results) - len(failed)}/{len(results)} przypadków OK"
          + (f" · nieudane: {failed}" if failed else ""))

    args.out.mkdir(parents=True, exist_ok=True)
    print("▶ zrzuty ekranu…")
    results = capture(results, args.out)
    (args.out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gallery = build_gallery(results, args.out)
    print(f"▶ galeria: {gallery}")
    if args.standalone:
        bundle = build_gallery(results, args.out, embed=True)
        size_mb = bundle.stat().st_size / 1_048_576
        print(f"▶ galeria w jednym pliku: {bundle} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
