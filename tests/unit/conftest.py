"""Shared fixtures for unit tests.

Tests live inside the alleasystent repo, so no sys.path manipulation is needed.
Run with: cd tests/unit && pytest  OR  pytest tests/unit/ from the repo root.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Clear lru_cache on get_settings() before each test."""
    from config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def minimal_env(monkeypatch):
    """Minimal environment variables required for Settings to instantiate."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret")
    return {
        "GOOGLE_API_KEY": "test-key",
        "JWT_SECRET": "test-jwt-secret",
    }
