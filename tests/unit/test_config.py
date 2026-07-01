"""Unit tests for config/settings.py."""
from __future__ import annotations

import pytest


class TestSettingsModelPool:
    def test_default_pool_when_empty(self, minimal_env):
        from config.settings import Settings
        s = Settings()
        pool = s.model_pool()
        assert isinstance(pool, list)
        assert len(pool) >= 1
        assert all(isinstance(m, str) for m in pool)

    def test_custom_pool_from_env(self, monkeypatch, minimal_env):
        monkeypatch.setenv("GEMINI_MODEL_POOL", "model-a,model-b")
        from config.settings import Settings
        s = Settings()
        assert s.model_pool() == ["model-a", "model-b"]

    def test_custom_pool_strips_whitespace(self, monkeypatch, minimal_env):
        monkeypatch.setenv("GEMINI_MODEL_POOL", " model-a , model-b ")
        from config.settings import Settings
        s = Settings()
        assert s.model_pool() == ["model-a", "model-b"]

    def test_fast_pool_default(self, minimal_env):
        from config.settings import Settings
        s = Settings()
        pool = s.model_fast_pool()
        assert isinstance(pool, list)
        assert len(pool) >= 1

    def test_fast_pool_custom(self, monkeypatch, minimal_env):
        monkeypatch.setenv("GEMINI_MODEL_FAST_POOL", "fast-1")
        from config.settings import Settings
        s = Settings()
        assert s.model_fast_pool() == ["fast-1"]


class TestSettingsIsProduction:
    def test_development_is_not_production(self, minimal_env):
        from config.settings import Settings
        s = Settings()
        assert s.is_production is False

    def test_production_flag(self, monkeypatch, minimal_env):
        monkeypatch.setenv("APP_ENV", "production")
        from config.settings import Settings
        s = Settings()
        assert s.is_production is True


class TestGetSettingsCache:
    def test_returns_same_instance(self, minimal_env):
        from config.settings import get_settings
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_cache_cleared_between_tests(self, minimal_env):
        from config.settings import get_settings
        s = get_settings()
        assert s is not None


class TestSettingsDefaults:
    def test_allegro_defaults(self, minimal_env):
        from config.settings import Settings
        s = Settings()
        assert s.allegro_api_url == "https://api.allegro.pl"
        assert s.allegro_token_store == "file"

    def test_redis_url_default_empty(self, minimal_env):
        from config.settings import Settings
        s = Settings()
        assert s.redis_url == ""

    def test_gcp_project_default_empty(self, minimal_env):
        from config.settings import Settings
        s = Settings()
        assert s.gcp_project_id == ""
