"""
Shared fixtures and helpers for AllEasystent API tests.

Configuration (environment variables or defaults):
  ALLEASYSTENT_URL   — base URL of the running backend, e.g. https://your-app.railway.app
                       default: http://localhost:8080
  ALLEGRO_AUTHED     — set to "1" to run tests that require Allegro authorization
"""

import os
import time
import uuid
import pytest
import httpx

BASE_URL = os.environ.get("ALLEASYSTENT_URL", "http://localhost:8080").rstrip("/")
ALLEGRO_AUTHED = os.environ.get("ALLEGRO_AUTHED", "0") == "1"

requires_allegro = pytest.mark.skipif(
    not ALLEGRO_AUTHED,
    reason="Wymaga autoryzacji Allegro. Ustaw ALLEGRO_AUTHED=1 po autoryzacji.",
)


def new_session() -> str:
    """Generate a unique session ID for test isolation."""
    return f"test_{uuid.uuid4().hex[:12]}"


def query(message: str, session_id: str | None = None, sender_id: str = "test_user") -> dict:
    """
    Send a message to POST /query and return the full response dict:
      {response: str, agent: str, sources: list[str]}
    """
    payload = {
        "message": message,
        "session_id": session_id or new_session(),
        "sender_id": sender_id,
    }
    resp = httpx.post(f"{BASE_URL}/query", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def allegro_auth_status() -> dict:
    resp = httpx.get(f"{BASE_URL}/allegro/auth/status", timeout=10)
    resp.raise_for_status()
    return resp.json()
