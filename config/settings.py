from __future__ import annotations

"""Central configuration via Pydantic settings — reads from .env and environment."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Google Gemini ────────────────────────────────────────────────────────
    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    gemini_model_fast: str = "gemini-2.0-flash"
    gemini_max_tokens: int = 16000
    # Comma-separated rotation pools; empty = auto-derive from the two models above.
    # Example: GEMINI_MODEL_POOL=gemini-2.5-flash,gemini-2.0-flash,gemini-1.5-flash
    gemini_model_pool: str = ""
    gemini_model_fast_pool: str = ""

    def model_pool(self) -> list[str]:
        """Ordered pool for agents that use gemini_model."""
        if self.gemini_model_pool:
            return [m.strip() for m in self.gemini_model_pool.split(",") if m.strip()]
        return list(dict.fromkeys([self.gemini_model, self.gemini_model_fast]))

    def model_fast_pool(self) -> list[str]:
        """Ordered pool for agents that use gemini_model_fast."""
        if self.gemini_model_fast_pool:
            return [m.strip() for m in self.gemini_model_fast_pool.split(",") if m.strip()]
        return list(dict.fromkeys([self.gemini_model_fast, self.gemini_model]))

    # ── Facebook Messenger ────────────────────────────────────────────────────
    facebook_page_access_token: str = ""
    facebook_verify_token: str = "alleasystent_verify_token"
    facebook_app_secret: str = ""

    # ── Allegro ───────────────────────────────────────────────────────────────
    allegro_client_id: str = "56eb5dd3b0ba4f6e82240aafd6b1c8dd"
    allegro_client_secret: str = "rCIyemc4Y1iFPwRiJefmzydhURbMlqv30dY9QuI516eOKAaoKXin3CPHrMvawIHu"
    allegro_redirect_uri: str = "http://localhost:8000/allegro/callback"
    allegro_api_url: str = "https://api.allegro.pl"
    allegro_auth_url: str = "https://allegro.pl/auth/oauth"
    # Store tokens in GCP Secret Manager in production; file path for dev
    allegro_token_store: Literal["file", "secret_manager"] = "file"
    allegro_token_file: str = ".allegro_tokens.json"

    # ── GCP ──────────────────────────────────────────────────────────────────
    gcp_project_id: str = ""
    gcp_region: str = "europe-central2"
    firestore_collection_conversations: str = "conversations"
    pubsub_topic_incoming: str = "incoming-messages"
    pubsub_topic_outgoing: str = "outgoing-messages"
    pubsub_subscription_incoming: str = "incoming-messages-sub"

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_backend: Literal["chromadb", "vertex_ai"] = "chromadb"
    chromadb_path: str = "./data/chromadb"
    vertex_ai_index_endpoint: str = ""
    vertex_ai_index_id: str = ""
    # Embedding model: local sentence-transformers or Vertex AI
    embedding_backend: Literal["local", "vertex_ai"] = "local"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_top_k: int = 5

    # ── Application ───────────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    port: int = 8080

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
