from __future__ import annotations

"""Lightweight per-request stage timing.

User queries were taking 30s-60s to answer and it wasn't clear from the
existing logs which stage was responsible (LLM call, Allegro API round-trip,
session store, ...). StageTimer wraps each stage of a request in a `with`
block and emits one summary log line per request breaking down wall-clock
time by stage, so a slow turn can be diagnosed straight from production logs.

Not a profiler (no CPU sampling) — just elapsed wall-clock time per named
stage, which is what matters here since every stage is I/O-bound (network
calls to the Gemini API, Allegro API, Redis).
"""

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("perf")


class StageTimer:
    """Accumulates named stage durations for one request and logs them as a
    single summary line. Stages can repeat (e.g. multiple tool calls) — each
    occurrence is recorded and totaled per name in the summary."""

    def __init__(self, label: str):
        self._label = label
        self._start = time.perf_counter()
        self._stages: list[tuple[str, float]] = []

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._stages.append((name, (time.perf_counter() - t0) * 1000))

    def log(self, **extra) -> None:
        total_ms = (time.perf_counter() - self._start) * 1000
        totals: dict[str, float] = {}
        for name, ms in self._stages:
            totals[name] = totals.get(name, 0.0) + ms
        breakdown = ", ".join(f"{name}={ms:.0f}ms" for name, ms in totals.items())
        extra_str = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        logger.info("PERF %s total=%.0fms %s [%s]", self._label, total_ms, extra_str, breakdown)
