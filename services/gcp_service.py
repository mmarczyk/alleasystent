from __future__ import annotations

"""GCP service layer: session storage (Redis/in-memory) and Pub/Sub (async messaging).

Storage priority for conversation sessions:
  1. Redis (if REDIS_URL is configured) — the real store in every deployment
     this app currently runs in, GCP included (an external Redis instance,
     not GCP's own).
  2. In-memory dict (local dev fallback, or Redis unreachable)

Used to also try Firestore first when GCP_PROJECT_ID was set, on the theory
that a GCP deployment would want a GCP-native store. In practice the actual
deployment already runs Redis regardless of host (GCP or not), so Firestore
was just a second, redundant persistence path adding a network round-trip
to every request without being useful for anything Redis wasn't already
doing — removed. Bring it back only if a deployment target genuinely has no
usable Redis but does have Firestore.
"""

import json
import logging
from datetime import datetime
from typing import Any

from config.settings import get_settings
from models.conversation import ChannelType, ConversationSession, MessageRole

logger = logging.getLogger(__name__)

_REDIS_SESSION_PREFIX = "conv:"
_REDIS_SESSION_TTL = 60 * 60 * 24 * 30  # 30 days


class SessionStore:
    """
    Manages conversation sessions.

    Redis-backed when REDIS_URL is configured, in-memory otherwise (local
    development, or Redis unreachable).
    """

    def __init__(self):
        self._settings = get_settings()
        self._redis = None
        self._memory_store: dict[str, dict] = {}
        self._init_redis()

    def _init_redis(self) -> None:
        redis_url = self._settings.redis_url
        if not redis_url or not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
            logger.info("REDIS_URL not set or invalid — using in-memory session store")
            return
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            logger.info("Redis client initialized for conversation sessions")
        except Exception as exc:
            logger.warning("Redis init failed (%s) — using in-memory session store", exc)

    # ── Session CRUD ─────────────────────────────────────────────────────────

    async def get_session(self, session_id: str) -> ConversationSession | None:
        if self._redis is not None:
            raw = await self._redis.get(f"{_REDIS_SESSION_PREFIX}{session_id}")
            if raw is None:
                return None
            return ConversationSession.model_validate(json.loads(raw))

        data = self._memory_store.get(session_id)
        if data is None:
            return None
        return ConversationSession.model_validate(data)

    async def save_session(self, session: ConversationSession) -> None:
        session.updated_at = datetime.utcnow()
        data = json.loads(session.model_dump_json())

        if self._redis is not None:
            await self._redis.set(
                f"{_REDIS_SESSION_PREFIX}{session.session_id}",
                json.dumps(data),
                ex=_REDIS_SESSION_TTL,
            )
            return

        self._memory_store[session.session_id] = data

    async def get_or_create_session(
        self,
        session_id: str,
        channel: ChannelType,
        sender_id: str,
    ) -> ConversationSession:
        existing = await self.get_session(session_id)
        if existing:
            return existing
        session = ConversationSession(
            session_id=session_id,
            channel=channel,
            sender_id=sender_id,
        )
        await self.save_session(session)
        return session

    async def list_sessions(
        self,
        channel: ChannelType | None = None,
        limit: int = 50,
    ) -> list[ConversationSession]:
        if self._redis is not None:
            keys = await self._redis.keys(f"{_REDIS_SESSION_PREFIX}*")
            sessions = []
            for key in keys[:limit]:
                raw = await self._redis.get(key)
                if raw:
                    s = ConversationSession.model_validate(json.loads(raw))
                    if channel is None or s.channel == channel:
                        sessions.append(s)
            return sessions[:limit]

        sessions = [ConversationSession.model_validate(v) for v in self._memory_store.values()]
        if channel:
            sessions = [s for s in sessions if s.channel == channel]
        return sessions[:limit]


class PubSubService:
    """
    Publishes messages to Pub/Sub topics for async processing.

    Falls back to direct processing when Pub/Sub is not configured.
    """

    def __init__(self):
        self._settings = get_settings()
        self._publisher = None
        self._init_publisher()

    def _init_publisher(self) -> None:
        if not self._settings.gcp_project_id:
            return
        try:
            from google.cloud import pubsub_v1

            self._publisher = pubsub_v1.PublisherClient()
            logger.info("Pub/Sub publisher initialized")
        except ImportError:
            logger.warning("google-cloud-pubsub not installed — Pub/Sub disabled")
        except Exception as exc:
            logger.warning("Pub/Sub init failed: %s", exc)

    def _topic_path(self, topic_name: str) -> str:
        return f"projects/{self._settings.gcp_project_id}/topics/{topic_name}"

    async def publish_incoming_message(self, payload: dict[str, Any]) -> str | None:
        """Publish a normalized incoming message for async agent processing."""
        return await self._publish(
            self._settings.pubsub_topic_incoming,
            payload,
            {"message_type": "incoming"},
        )

    async def publish_outgoing_message(self, payload: dict[str, Any]) -> str | None:
        """Publish an outgoing message for async delivery."""
        return await self._publish(
            self._settings.pubsub_topic_outgoing,
            payload,
            {"message_type": "outgoing"},
        )

    async def _publish(
        self,
        topic_name: str,
        data: dict[str, Any],
        attributes: dict[str, str] | None = None,
    ) -> str | None:
        if self._publisher is None:
            logger.debug("Pub/Sub not available — skipping publish to %s", topic_name)
            return None
        try:
            future = self._publisher.publish(
                self._topic_path(topic_name),
                json.dumps(data).encode("utf-8"),
                **(attributes or {}),
            )
            message_id = future.result(timeout=10)
            logger.debug("Published to %s: %s", topic_name, message_id)
            return message_id
        except Exception as exc:
            logger.error("Pub/Sub publish failed: %s", exc)
            return None
