"""Unit tests for webhooks/facebook_webhook.py using FastAPI TestClient."""
from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


APP_SECRET = "test-secret"
VERIFY_TOKEN = "test-verify-token"


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("FACEBOOK_APP_SECRET", APP_SECRET)
    monkeypatch.setenv("FACEBOOK_VERIFY_TOKEN", VERIFY_TOKEN)


@pytest.fixture()
def app():
    """Build a fresh FastAPI app with the facebook webhook router."""
    import webhooks.facebook_webhook as wh_module
    # Reset singletons
    wh_module._communication_agent = None
    wh_module._orchestrator = None

    application = FastAPI()
    application.include_router(wh_module.router)
    yield application

    # Cleanup
    wh_module._communication_agent = None
    wh_module._orchestrator = None


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=True)


def _make_sig(payload: bytes) -> str:
    digest = hmac.new(APP_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class TestVerifyWebhook:
    def test_valid_token_returns_challenge(self, client):
        resp = client.get(
            "/webhook/facebook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "my_challenge_123",
            },
        )
        assert resp.status_code == 200
        assert resp.text == "my_challenge_123"

    def test_wrong_token_returns_403(self, client):
        resp = client.get(
            "/webhook/facebook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong-token",
                "hub.challenge": "challenge",
            },
        )
        assert resp.status_code == 403

    def test_wrong_mode_returns_403(self, client):
        resp = client.get(
            "/webhook/facebook",
            params={
                "hub.mode": "unsubscribe",
                "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "challenge",
            },
        )
        assert resp.status_code == 403

    def test_missing_params_returns_403(self, client):
        resp = client.get("/webhook/facebook")
        assert resp.status_code == 403


class TestReceiveMessage:
    def _page_payload(self, sender_id: str = "user-1", text: str = "hello"):
        return {
            "object": "page",
            "entry": [{
                "id": "page-1",
                "messaging": [{
                    "sender": {"id": sender_id},
                    "message": {"text": text},
                }]
            }]
        }

    def _post(self, client, payload: dict, sign: bool = True):
        body = json.dumps(payload).encode()
        headers = {}
        if sign:
            headers["X-Hub-Signature-256"] = _make_sig(body)
        return client.post(
            "/webhook/facebook",
            content=body,
            headers={"Content-Type": "application/json", **headers},
        )

    def test_valid_page_event_returns_ok(self, client):
        with patch("webhooks.facebook_webhook.get_communication_agent") as mock_ca, \
             patch("webhooks.facebook_webhook.get_orchestrator") as mock_orc:
            mock_ca.return_value._fb.verify_webhook_signature.return_value = True
            resp = self._post(client, self._page_payload())
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_non_page_object_returns_ignored(self, client):
        payload = {"object": "user", "entry": []}
        body = json.dumps(payload).encode()
        with patch("webhooks.facebook_webhook.get_communication_agent") as mock_ca:
            mock_ca.return_value._fb.verify_webhook_signature.return_value = True
            resp = client.post(
                "/webhook/facebook",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": _make_sig(body),
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_invalid_signature_returns_403(self, client, monkeypatch):
        payload = self._page_payload()
        body = json.dumps(payload).encode()
        with patch("webhooks.facebook_webhook.get_communication_agent") as mock_ca:
            mock_ca.return_value._fb.verify_webhook_signature.return_value = False
            resp = client.post(
                "/webhook/facebook",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": "sha256=wrong",
                },
            )
        assert resp.status_code == 403

    def test_invalid_json_returns_400(self, client):
        bad_body = b"not json"
        with patch("webhooks.facebook_webhook.get_communication_agent") as mock_ca:
            mock_ca.return_value._fb.verify_webhook_signature.return_value = True
            resp = client.post(
                "/webhook/facebook",
                content=bad_body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": _make_sig(bad_body),
                },
            )
        assert resp.status_code == 400

    def test_no_app_secret_skips_signature_check(self, client, monkeypatch):
        monkeypatch.setenv("FACEBOOK_APP_SECRET", "")
        from config.settings import get_settings
        get_settings.cache_clear()
        payload = self._page_payload()
        body = json.dumps(payload).encode()
        with patch("webhooks.facebook_webhook.get_communication_agent") as mock_ca, \
             patch("webhooks.facebook_webhook.get_orchestrator"):
            mock_ca.return_value._fb.verify_webhook_signature.return_value = True
            resp = client.post(
                "/webhook/facebook",
                content=body,
                headers={"Content-Type": "application/json"},
            )
        # No signature header but no secret — should not 403
        assert resp.status_code == 200
