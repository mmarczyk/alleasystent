from __future__ import annotations

"""UI regression tests for the full-screen document viewer.

Drives the real PWA (web/index.html + web/js/app.js) in Chromium against the
same stub backend the screenshot capture uses, so these assert on what the
browser actually renders — not on a re-implementation of the rules.

Skipped when Playwright or a Chromium build isn't available.
"""

import json

import pytest

pytest.importorskip("playwright.sync_api", reason="playwright not installed")

from playwright.sync_api import sync_playwright  # noqa: E402

from tests.tools.capture_screenshots import (  # noqa: E402
    CHROMIUM_PATH,
    _State,
    _fake_jwt,
    start_stub_server,
)

# A document reply exactly as the model is told to shape it for
# get_order_details: a ```summary block, then the full document.
DOC_REPLY = """```summary
- Zamówienie: `0c4854a0-9646-11f1-8028-338c43adc37a`
- Kupujący: anna.kowalska88
- Status: Nowe
- Wartość: 429,98 PLN
```
# Szczegóły zamówienia 0c4854a0

## Produkty
- Ekspres do kawy DeLonghi Magnifica S ECAM — 1 × 399,99 PLN

## Dostawa
Allegro Paczkomaty InPost, POZ01A.
"""

# A table reply as _OUTPUT_FORMAT_INSTRUCTIONS["table"] asks for it: the table
# first, one or two summary sentences after it.
TABLE_REPLY = """| Zamówienie | Kupujący | Wartość |
| --- | --- | --- |
| 0c4854a0 | anna.kowalska88 | 429,98 PLN |
| 1d5965b1 | marek_zielinski | 899,00 PLN |

Łącznie 2 zamówienia o wartości 1328,98 PLN. Jedno jest już po terminie nadania.
"""

# Prose already leads — nothing may be reordered.
PROSE_FIRST_REPLY = """## Podsumowanie sprzedaży

Sprzedaż w tym miesiącu wyniosła **3049,66 PLN** z 6 zamówień, przy średniej
wartości zamówienia 508,28 PLN. Największy udział mają ekspresy do kawy.

## Top produkty

| Produkt | Przychód |
| --- | --- |
| Ekspres do kawy DeLonghi Magnifica S ECAM | 1648,99 PLN |
| Odkurzacz pionowy Dyson V12 Detect Slim | 899,00 PLN |
| Blender kielichowy Philips HR2291 | 259,00 PLN |
"""

BUTTON_REPLY = TABLE_REPLY + (
    '\n<button class="btn-monitoring" onclick="OrderMonitor.enable()">'
    "🔔 Włącz automatyczne sprawdzanie</button>"
)


class _Ui:
    """Sends one reply through the real chat and reports what the doc viewer shows."""

    def __init__(self, page, port, state):
        self._page = page
        self._port = port
        self._state = state

    def show(self, reply: str, fmt: str) -> dict:
        page = self._page
        page.request.post(
            f"http://127.0.0.1:{self._port}/__case",
            data=json.dumps({"text": reply, "format": fmt}),
        )
        page.goto(f"http://127.0.0.1:{self._port}/index.html", wait_until="networkidle")
        page.wait_for_selector("#user-input", state="visible", timeout=15000)
        page.fill("#user-input", "pokaż")
        page.click("#btn-send")
        page.wait_for_selector("#waiting-bubble", state="detached", timeout=20000)
        page.wait_for_timeout(250)
        return page.evaluate(
            """() => {
                const viewer = document.getElementById('doc-viewer');
                const content = document.getElementById('doc-content');
                const tagOrder = [...content.children].map(el => el.tagName);
                const table = content.querySelector('table');
                const rule = content.querySelector('hr');
                const idx = (el) => el ? [...content.children].indexOf(
                    el.closest('#doc-content > *')) : -1;
                return {
                    open: !viewer.classList.contains('hidden'),
                    text: content.innerText.replace(/\\s+/g, ' ').trim(),
                    tagOrder,
                    hasPre: !!content.querySelector('pre'),
                    tableIdx: idx(table),
                    ruleIdx: idx(rule),
                    buttonIdx: idx(content.querySelector('button')),
                    childCount: content.children.length,
                };
            }"""
        )


@pytest.fixture(scope="module")
def ui():
    state = _State()
    server, port = start_stub_server(state)
    with sync_playwright() as pw:
        kwargs = {}
        if CHROMIUM_PATH.exists():
            kwargs["executable_path"] = str(CHROMIUM_PATH)
        try:
            browser = pw.chromium.launch(**kwargs)
        except Exception as exc:  # noqa: BLE001
            server.shutdown()
            pytest.skip(f"no usable Chromium: {exc}")
        context = browser.new_context(viewport={"width": 1024, "height": 800},
                                      locale="pl-PL", timezone_id="Europe/Warsaw")
        context.route("**/sw.js", lambda route: route.abort())
        context.add_init_script(
            f"try {{ localStorage.setItem('ae_session_token', {json.dumps(_fake_jwt())}); }} catch (e) {{}}"
        )
        page = context.new_page()
        try:
            yield _Ui(page, port, state)
        finally:
            context.close()
            browser.close()
            server.shutdown()


def test_summary_block_leads_the_document(ui):
    view = ui.show(DOC_REPLY, "document")
    assert view["open"], "document reply should open the full-screen viewer"
    # The summary is the first thing on the big screen, above the document.
    assert view["text"].startswith("Zamówienie: 0c4854a0")
    assert view["text"].index("Kupujący: anna.kowalska88") < view["text"].index("Szczegóły zamówienia")
    # …as rendered markdown, never as a raw ```summary code block.
    assert not view["hasPre"], "summary block leaked into the doc view as a code block"
    assert view["ruleIdx"] > -1, "no rule separating the summary from the document"


def test_table_summary_sentence_moves_above_the_table(ui):
    view = ui.show(TABLE_REPLY, "table")
    assert view["open"]
    assert view["text"].startswith("Łącznie 2 zamówienia")
    assert 0 <= view["ruleIdx"] < view["tableIdx"], view["tagOrder"]


def test_reply_that_already_leads_with_prose_is_left_alone(ui):
    view = ui.show(PROSE_FIRST_REPLY, "dashboard")
    assert view["open"]
    assert view["text"].startswith("Podsumowanie sprzedaży")
    assert view["ruleIdx"] == -1, "nothing was reordered, so no rule should be added"


def test_action_button_stays_at_the_bottom(ui):
    view = ui.show(BUTTON_REPLY, "table")
    assert view["open"]
    assert view["buttonIdx"] == view["childCount"] - 1, view["tagOrder"]
    assert view["buttonIdx"] > view["tableIdx"]
