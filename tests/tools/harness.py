from __future__ import annotations

"""Wiring that runs a real AllegroAgent tool call against the fake APIs.

The agent, the Allegro service, the inFakt service and every formatter are the
production ones — only the HTTP transports and the Redis-backed monitoring
flags are swapped out. What a tool returns here is byte-for-byte what it would
return against a live store holding tests/tools/dataset.py's data.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Settings are read at import time of the services, so the environment has to be
# complete before anything below is imported.
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-used")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("ALLEGRO_MOCK_TOKEN", "fake-allegro-access-token")
os.environ.setdefault("ALLEGRO_TOKEN_FILE", "/nonexistent/.allegro_tokens.json")
os.environ.setdefault("INFAKT_API_KEY", "test-infakt-key")
os.environ.setdefault("APP_ENV", "development")

import httpx  # noqa: E402

from tests.tools.fake_allegro import (  # noqa: E402
    FakeAllegroAPI,
    FakeInfaktAPI,
    allegro_transport,
    infakt_transport,
)

USER_ID = "demo_seller"

# Monitor kinds as stored by services/monitor_state.py, keyed by the short name
# used in the case matrix.
MONITOR_KINDS = {
    "message": "message",
    "message_reminder": "message_reminder",
    "invoice_reminder": "invoice_reminder",
    "returns": "returns_complaints",
}

ALL_MONITORS_OFF: dict[str, bool] = {
    "order": False, "message": False, "message_reminder": False,
    "invoice_reminder": False, "returns": False,
}


class ToolHarness:
    def __init__(self, agent, allegro_api: FakeAllegroAPI, infakt_api: FakeInfaktAPI):
        self.agent = agent
        self.allegro_api = allegro_api
        self.infakt_api = infakt_api

    async def run(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Run one tool exactly the way the agent's tool loop does."""
        return await self.agent._execute_tool(tool_name, tool_input)


@asynccontextmanager
async def tool_harness(
    monitors: dict[str, bool] | None = None,
    empty: bool = False,
    trim_listed_phones: bool = False,
):
    """Yield a ToolHarness with fake transports and monitoring flags installed.

    `trim_listed_phones` serves order LISTINGS without phone numbers (the
    single-order endpoint keeps them) — see FakeAllegroAPI for why that shape
    is worth being able to reproduce."""
    from config.settings import get_settings
    from services import monitor_state, order_monitor
    from services.allegro_service import AllegroService
    from services.infakt_service import InfaktService

    get_settings.cache_clear()
    settings = get_settings()
    flags = {**ALL_MONITORS_OFF, **(monitors or {})}

    allegro_api = FakeAllegroAPI(empty=empty, trim_listed_phones=trim_listed_phones)
    infakt_api = FakeInfaktAPI()

    allegro = AllegroService(user_id=USER_ID)
    await allegro._client.aclose()
    allegro._client = httpx.AsyncClient(
        base_url=settings.allegro_api_url,
        transport=allegro_transport(allegro_api),
    )

    infakt = InfaktService()
    await infakt._client.aclose()
    infakt._client = httpx.AsyncClient(
        base_url=settings.infakt_api_url,
        transport=infakt_transport(infakt_api),
    )
    prev_infakt_instance = InfaktService._instance
    InfaktService._instance = infakt

    # Redis-backed monitoring flags → in-memory booleans.
    async def fake_order_enabled(user_id: str) -> bool:
        return flags["order"]

    async def fake_kind_enabled(kind: str, user_id: str) -> bool:
        for short, full in MONITOR_KINDS.items():
            if full == kind:
                return flags[short]
        return False

    prev_order = order_monitor.is_monitor_enabled
    prev_kind = monitor_state.is_monitor_enabled
    order_monitor.is_monitor_enabled = fake_order_enabled
    monitor_state.is_monitor_enabled = fake_kind_enabled

    # AllegroAgent resolves its service through the singleton registry — seed it
    # so the agent picks up the fake-transport instance instead of building a
    # real one that would try to reach api.allegro.pl.
    prev_instances = dict(AllegroService._instances)
    AllegroService._instances[USER_ID] = allegro

    from agents.allegro.allegro_agent import AllegroAgent

    agent = AllegroAgent(user_id=USER_ID)

    try:
        yield ToolHarness(agent, allegro_api, infakt_api)
    finally:
        order_monitor.is_monitor_enabled = prev_order
        monitor_state.is_monitor_enabled = prev_kind
        InfaktService._instance = prev_infakt_instance
        AllegroService._instances = prev_instances
        await allegro._client.aclose()
        await infakt._client.aclose()
