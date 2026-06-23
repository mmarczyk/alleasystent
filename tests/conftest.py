"""
Shared fixtures and helpers for AllEasystent API tests.

Configuration (environment variables or defaults):
  ALLEASYSTENT_URL    — base URL of the running backend, e.g. https://your-app.railway.app
                        default: http://localhost:8080
  ALLEGRO_AUTHED      — set to "1" to run tests requiring Allegro access.
                        Automatically true when ALLEGRO_MOCK_MODE=1 (mock server).
  ALLEGRO_MOCK_MODE   — set to "1" when pointing at the mock Allegro server.
                        Implies ALLEGRO_AUTHED=1 without real credentials.
  JWT_SECRET          — must match the main app's JWT_SECRET to generate a test session cookie
  TEST_DELAY_SECONDS  — seconds to wait between tests to avoid Gemini 429 rate limits (default: 3)
"""

import os
import time
import uuid
import pytest
import httpx

BASE_URL = os.environ.get("ALLEASYSTENT_URL", "http://localhost:8080").rstrip("/")

# Mock mode counts as "authed" — the mock server accepts any token
_MOCK_MODE = os.environ.get("ALLEGRO_MOCK_MODE", "0") == "1"
ALLEGRO_AUTHED = _MOCK_MODE or os.environ.get("ALLEGRO_AUTHED", "0") == "1"

# Delay between tests to stay within Gemini API rate limits
_TEST_DELAY = float(os.environ.get("TEST_DELAY_SECONDS", "3"))

requires_allegro = pytest.mark.skipif(
    not ALLEGRO_AUTHED,
    reason=(
        "Wymaga autoryzacji Allegro. "
        "Ustaw ALLEGRO_AUTHED=1 (prawdziwe) lub ALLEGRO_MOCK_MODE=1 (mock server)."
    ),
)


@pytest.fixture(autouse=True)
def _rate_limit_delay():
    """Pause between tests to avoid Gemini 429 Too Many Requests errors."""
    yield
    if _TEST_DELAY > 0:
        time.sleep(_TEST_DELAY)


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
    cookies = {"session": _SESSION_COOKIE} if _SESSION_COOKIE else {}
    resp = httpx.post(f"{BASE_URL}/query", json=payload, cookies=cookies, timeout=120)
    resp.raise_for_status()
    return resp.json()


def allegro_auth_status() -> dict:
    """
    Sprawdza status autoryzacji Allegro przez GET /auth/me.
    Zwraca {"status": "authorized", "authenticated": True} lub {"status": "idle"}.
    """
    cookies = {"session": _SESSION_COOKIE} if _SESSION_COOKIE else {}
    try:
        resp = httpx.get(f"{BASE_URL}/auth/me", cookies=cookies, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {"status": "authorized", "authenticated": True, **data}
        return {"status": "idle", "authenticated": False}
    except Exception:
        return {"status": "idle", "authenticated": False}
