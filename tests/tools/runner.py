from __future__ import annotations

"""Runs the whole tool matrix and reports what each tool returned.

    python -m tests.tools.runner            # human-readable summary
    python -m tests.tools.runner --json out.json

The JSON dump is what the screenshot capture (capture_screenshots.py) replays
through the real PWA, so both always show exactly the same tool output.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.tools.cases import CASES, Case  # noqa: E402
from tests.tools.harness import tool_harness  # noqa: E402


async def run_case(case: Case) -> dict:
    """Execute one case against a freshly wired harness (no cross-case caching)."""
    from agents.allegro.allegro_tools import resolve_output_format

    started = time.perf_counter()
    error = None
    output = ""
    api_calls: list[str] = []
    try:
        async with tool_harness(monitors=case.monitors, empty=case.empty) as h:
            output = await h.run(case.tool, dict(case.args))
            api_calls = [f"{m} {p}" for m, p, _ in h.allegro_api.calls]
            api_calls += [f"{m} inFakt{p}" for m, p in h.infakt_api.calls]
    except Exception as exc:  # noqa: BLE001 — the report wants the failure, not a traceback
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    missing = [s for s in case.expect if s not in output]
    return {
        "id": case.id,
        "tool": case.tool,
        "args": case.args,
        "question": case.question,
        "note": case.note,
        "monitors": case.monitors,
        "empty_store": case.empty,
        "no_api": case.no_api,
        "format": resolve_output_format([case.tool]),
        "output": output,
        "chars": len(output),
        "duration_ms": elapsed_ms,
        "api_calls": api_calls,
        "error": error,
        "missing_expectations": missing,
        "ok": error is None and not missing and bool(output.strip()),
    }


async def run_all(cases: list[Case] | None = None) -> list[dict]:
    results = []
    for case in cases or CASES:
        results.append(await run_case(case))
    return results


def _print_summary(results: list[dict]) -> None:
    width = max(len(r["id"]) for r in results) + 2
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        detail = ""
        if r["error"]:
            detail = f"  ← {r['error']}"
        elif r["missing_expectations"]:
            detail = f"  ← brak w odpowiedzi: {r['missing_expectations']}"
        print(f"[{mark}] {r['id']:<{width}} {r['format']:<9} "
              f"{r['chars']:>6} zn.  {r['duration_ms']:>7} ms{detail}")
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} przypadków OK")
    if failed:
        print("Nieudane: " + ", ".join(r["id"] for r in failed))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Allegro tool matrix")
    parser.add_argument("--json", type=Path, help="write full results to this JSON file")
    parser.add_argument("--only", nargs="*", help="run only these case ids")
    args = parser.parse_args()

    cases = CASES
    if args.only:
        wanted = set(args.only)
        cases = [c for c in CASES if c.id in wanted]

    results = asyncio.run(run_all(cases))
    _print_summary(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nJSON: {args.json}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
