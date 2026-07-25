from __future__ import annotations

"""
Entrypoint for the alleasystent-order-monitor Cloud Run Job.
Run as: python -m jobs.order_monitor_job (from the repo root, so the
services/config/models packages resolve — see Dockerfile.jobs).

Runs one polling pass over every user with automatic order checking enabled,
then exits — triggered on a schedule by Cloud Scheduler (see
deployment/setup_gcp.sh). Deliberately standalone: only imports the modules
it needs (services/order_monitor.py and its dependencies), not main.py, so
this ships in the slim jobs image (requirements-jobs.txt) instead of the full
web app image with its RAG/ML dependencies.
"""

import asyncio
import logging
import os

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from services.order_monitor import run_once  # noqa: E402

if __name__ == "__main__":
    asyncio.run(run_once())
