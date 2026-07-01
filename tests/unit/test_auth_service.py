"""Unit tests for services/auth_service.py."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest


SECRET = "test-jwt-secret"


@pytest.fixture(autouse=True)
def set_jwt_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", SECRET)


class TestCreateAndDecodeToken:
    def test_round_trip(self):
        from services.auth_service import create_session_token, decode_session_token
        user = {"sub": "user123", "name": "Test User"}
        token = create_session_token(user)
        decoded = decode_session_token(token)
        assert decoded["sub"] == "user123"
        assert decoded["name"] == "Test User"

    def test_sub_only(self):
        from services.auth_service import create_session_token, decode_session_token
        user = {"sub": "u1"}
        token = create_session_token(user)
        decoded = decode_session_token(token)
        assert decoded["sub"] == "u1"
        assert decoded["name"] == "u1"

    def test_token_contains_exp(self):
        from services.auth_service import create_session_token
        token = create_session_token({"sub": "u1"})
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        assert "exp" in payload
        assert payload["exp"] > time.time()

    def test_expired_token_raises(self):
        payload = {
            "sub": "u1",
            "name": "u1",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
        }
        token = jwt.encode(payload, SECRET, algorithm="HS256")
        from services.auth_service import decode_session_token
        with pytest.raises(jwt.ExpiredSignatureError):
            decode_session_token(token)

    def test_invalid_token_raises(self):
        from services.auth_service import decode_session_token
        with pytest.raises(jwt.InvalidTokenError):
            decode_session_token("not.a.valid.jwt")

    def test_wrong_secret_raises(self):
        payload = {"sub": "u1", "name": "u1",
                   "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
        token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
        from services.auth_service import decode_session_token
        with pytest.raises(jwt.InvalidTokenError):
            decode_session_token(token)


class TestSecretFunction:
    def test_raises_when_jwt_secret_empty(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "")
        from config.settings import get_settings
        get_settings.cache_clear()
        from services.auth_service import _secret
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            _secret()

    def test_returns_secret(self):
        from services.auth_service import _secret
        assert _secret() == SECRET


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_no_cookie_raises_401(self):
        from fastapi import HTTPException
        from services.auth_service import get_current_user
        request = MagicMock()
        request.cookies.get.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_cookie_returns_user(self):
        from services.auth_service import create_session_token, get_current_user
        token = create_session_token({"sub": "u1", "name": "Alice"})
        request = MagicMock()
        request.cookies.get.return_value = token
        user = await get_current_user(request)
        assert user["sub"] == "u1"

    @pytest.mark.asyncio
    async def test_expired_cookie_raises_401(self):
        from fastapi import HTTPException
        from services.auth_service import get_current_user
        payload = {
            "sub": "u1",
            "name": "u1",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
        }
        token = jwt.encode(payload, SECRET, algorithm="HS256")
        request = MagicMock()
        request.cookies.get.return_value = token
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_invalid_cookie_raises_401(self):
        from fastapi import HTTPException
        from services.auth_service import get_current_user
        request = MagicMock()
        request.cookies.get.return_value = "bad-token"
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request)
        assert exc_info.value.status_code == 401
