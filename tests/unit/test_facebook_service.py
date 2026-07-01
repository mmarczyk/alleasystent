"""Unit tests for services/facebook_service.py."""
from __future__ import annotations

import hashlib
import hmac

import pytest


APP_SECRET = "test-app-secret"


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("FACEBOOK_APP_SECRET", APP_SECRET)


def _make_signature(payload: bytes, secret: str) -> str:
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={expected}"


class TestVerifyWebhookSignature:
    def _make_service(self):
        from services.facebook_service import FacebookService
        return FacebookService()

    def test_valid_signature(self):
        svc = self._make_service()
        payload = b'{"test": "data"}'
        sig = _make_signature(payload, APP_SECRET)
        assert svc.verify_webhook_signature(payload, sig) is True

    def test_invalid_signature(self):
        svc = self._make_service()
        payload = b'{"test": "data"}'
        assert svc.verify_webhook_signature(payload, "sha256=wrong") is False

    def test_missing_sha256_prefix(self):
        svc = self._make_service()
        payload = b"data"
        sig = hmac.new(APP_SECRET.encode(), payload, hashlib.sha256).hexdigest()
        assert svc.verify_webhook_signature(payload, sig) is False

    def test_empty_payload(self):
        svc = self._make_service()
        payload = b""
        sig = _make_signature(payload, APP_SECRET)
        assert svc.verify_webhook_signature(payload, sig) is True

    def test_wrong_secret(self):
        svc = self._make_service()
        payload = b"data"
        sig = _make_signature(payload, "wrong-secret")
        assert svc.verify_webhook_signature(payload, sig) is False

    def test_tampered_payload(self):
        svc = self._make_service()
        payload = b"original"
        sig = _make_signature(payload, APP_SECRET)
        assert svc.verify_webhook_signature(b"tampered", sig) is False
