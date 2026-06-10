from __future__ import annotations

"""JWT session helpers — sessions are established via Allegro OAuth2 login."""

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import HTTPException, Request

from config.settings import get_settings


def _secret() -> str:
    s = get_settings().jwt_secret
    if not s:
        raise RuntimeError("JWT_SECRET env var must be set in production")
    return s


def create_session_token(user: dict[str, Any]) -> str:
    settings = get_settings()
    payload = {
        "sub": user["sub"],
        "name": user.get("name", user["sub"]),
        "exp": datetime.now(timezone.utc) + timedelta(days=settings.jwt_expire_days),
    }
    return jwt.encode(payload, _secret(), algorithm="HS256")


def decode_session_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, _secret(), algorithms=["HS256"])


async def get_current_user(request: Request) -> dict[str, Any]:
    token = request.cookies.get("session")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return decode_session_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session")
