"""
Shared fixtures and helpers for AllEasystent API tests.

Configuration (environment variables or defaults):
  ALLEASYSTENT_URL   — base URL of the running backend, e.g. https://your-app.railway.app
                       default: http://localhost:8080
  ALLEGRO_AUTHED     — set to "1" to run tests that require Allegro authorization
  JWT_SECRET         — must match the main app's JWT_SECRET to generate a test session cookie
"""

import os
import time
import pytest
import httpx

BASE_URL = os.environ.get("ALLEASYSTENT_URL", "http://localhost:8080").rstrip("/")
ALLEGRO_AUTHED = os.environ.get("ALLEGRO_AUTHED", "0") == "1"

requires_allegro = pytest.mark.skipif(
    not ALLEGRO_AUTHED,
    reason="Wymaga autoryzacji Allegro. Ustaw ALLEGRO_AUTHED=1 po autoryzacji.",
)


def _make_session_cookie() -> str | None:
    """Generate a test JWT session cookie if JWT_SECRET is available."""
    secret = os.environ.get("JWT_SECRET", "")
    if not secret:
        return None
    try:
        from datetime import datetime, timedelta, timezone
        import jwt
        payload = {
            "sub": "test_user",
            "name": "Test User",
            "exp": datetime.now(timezone.utc) + timedelta(days=1),
        }
        return jwt.encode(payload, secret, algorithm="HS256")
    except Exception:
        return None


_SESSION_COOKIE = _make_session_cookie()


def new_session() -> str:
    """Generate a unique session ID for test isolation."""
    return f"test_{int(time.time() * 1000)}"


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
    cookies = {"session": _SESSION_COOKIE} if _SESSION_COOKIE else {}
    resp = httpx.post(f"{BASE_URL}/query", json=payload, cookies=cookies, timeout=60)
    resp.raise_for_status()
    return resp.json()


def allegro_auth_status() -> dict:
    resp = httpx.get(f"{BASE_URL}/allegro/auth/status", timeout=10)
    resp.raise_for_status()
    return resp.json()
