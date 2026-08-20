from __future__ import annotations

"""
HTTP entrypoint for the alleasystent-order-monitor Cloud Run *service*.
Run as: python -m jobs.order_monitor_service (from the repo root, so the
services/config/models packages resolve — see Dockerfile.jobs).

Deployed as a small always-there Cloud Run service, POSTed to by Cloud
Scheduler every couple of minutes (see deployment/setup_gcp.sh), rather than
as a Cloud Run Job. Cloud Run Jobs cold-start a brand-new container for every
single execution with no warm-instance reuse between runs — at a 2-minute
schedule that made cold-start overhead (image pull, interpreter start,
imports) the dominant cost for what's otherwise a sub-second poll. A service
can reuse a warm instance across back-to-back scheduler ticks like any other
Cloud Run traffic, so the same cadence costs a fraction as much.

Auth is handled by the Cloud Run platform (deployed with
--no-allow-unauthenticated; Cloud Scheduler calls in with an OIDC identity
token from a service account holding roles/run.invoker) — requests that
reach this process are already authorized, no token check needed here.

Only stdlib is used (http.server) to keep requirements-jobs.txt free of a
web framework dependency for a single route.
"""

import asyncio
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from services.order_monitor import run_once as run_order_monitor  # noqa: E402
from services.return_complaint_monitor import run_once as run_returns_complaints_monitor  # noqa: E402


async def _run_all() -> None:
    # Isolated try/except per monitor so one failing (e.g. a transient Allegro
    # API error surfacing unexpectedly) doesn't block the other from running.
    try:
        await run_order_monitor()
    except Exception:
        logger.exception("order monitor pass failed")
    try:
        await run_returns_complaints_monitor()
    except Exception:
        logger.exception("returns/complaints monitor pass failed")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        logger.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/run":
            self._respond(404, {"error": "not found"})
            return
        try:
            asyncio.run(_run_all())
            self._respond(200, {"status": "ok"})
        except Exception:
            logger.exception("order monitor run failed")
            self._respond(500, {"status": "error"})

    def _respond(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    logger.info("order-monitor service listening on :%d", port)
    server.serve_forever()
