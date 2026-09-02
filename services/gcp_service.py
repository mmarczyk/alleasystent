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
import re
from datetime import datetime
from typing import Any

from config.settings import get_settings
from models.conversation import ChannelType, ConversationSession, MessageRole

logger = logging.getLogger(__name__)

_REDIS_SESSION_PREFIX = "conv:"
# One sorted set per user listing that user's conversations, scored by last
# update. Redis has no cheap "every key for this user" query (KEYS/SCAN is
# O(keyspace) and blocks the server), so the index is what makes listing a
# seller's chats on a second device an O(limit) read instead.
_REDIS_USER_INDEX_PREFIX = "convidx:"
_REDIS_SESSION_TTL = 60 * 60 * 24 * 30  # 30 days

# Users whose id is a plain identifier can have their conversations recovered
# with a targeted SCAN — see SessionStore._scan_user_sessions.
_SAFE_USER_ID_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")


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
        # Users this process has already run the index-recovery scan for — see
        # _scan_user_sessions. Without it, a brand-new seller (no conversations,
        # so an empty index) would trigger a keyspace scan on every poll of
        # their chat list.
        self._index_scanned: set[str] = set()
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
            await self._index_session(session)
            return

        self._memory_store[session.session_id] = data

    async def _index_session(self, session: ConversationSession) -> None:
        """Record the session in its owner's conversation index.

        Never fails a save: the index is a lookup convenience, and a session
        missing from it is still readable by id (and re-indexed by its next
        save), while a raised error here would lose the turn just written.
        """
        if self._redis is None or not session.sender_id:
            return
        try:
            key = f"{_REDIS_USER_INDEX_PREFIX}{session.sender_id}"
            await self._redis.zadd(key, {session.session_id: session.updated_at.timestamp()})
            await self._redis.expire(key, _REDIS_SESSION_TTL)
        except Exception as exc:
            logger.warning("Could not index session %s: %s", session.session_id, exc)

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

    # ── Per-user conversation list (cross-device chat sync) ───────────────────

    async def list_user_sessions(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[ConversationSession]:
        """Every conversation belonging to one user, most recently updated first.

        This is what lets the seller open a thread started on their desktop from
        their phone: conversations are stored per user, not per device, and the
        device only holds a cache of them.
        """
        if self._redis is None:
            sessions = [
                ConversationSession.model_validate(v) for v in self._memory_store.values()
            ]
            sessions = [s for s in sessions if s.sender_id == user_id]
            sessions.sort(key=lambda s: s.updated_at, reverse=True)
            return sessions[:limit]

        key = f"{_REDIS_USER_INDEX_PREFIX}{user_id}"
        session_ids = await self._redis.zrevrange(key, 0, limit - 1)
        if not session_ids:
            return await self._scan_user_sessions(user_id, limit)

        sessions: list[ConversationSession] = []
        expired: list[str] = []
        for session_id in session_ids:
            raw = await self._redis.get(f"{_REDIS_SESSION_PREFIX}{session_id}")
            if raw is None:
                # The session's own key hit its 30-day TTL while the index entry
                # (refreshed on every save of any of this user's chats) outlived
                # it — drop the dangling reference.
                expired.append(session_id)
                continue
            sessions.append(ConversationSession.model_validate(json.loads(raw)))
        if expired:
            await self._redis.zrem(key, *expired)
        return sessions

    async def _scan_user_sessions(
        self,
        user_id: str,
        limit: int,
    ) -> list[ConversationSession]:
        """Recover a user's conversations when the index has no entry for them.

        Only reachable for a user whose chats all predate the index (it is
        written on every save, so one new turn is enough to populate it), and
        only for sessions keyed `<user>:<conversation>` the way the chat API
        keys them. The SCAN is bounded by the user prefix, the result is
        written back to the index, and it runs at most once per user per
        process — an empty index is also what a user with no chats at all
        looks like, and their app polls this list every few seconds.
        """
        if not _SAFE_USER_ID_RE.match(user_id) or user_id in self._index_scanned:
            return []
        self._index_scanned.add(user_id)
        sessions: list[ConversationSession] = []
        try:
            pattern = f"{_REDIS_SESSION_PREFIX}{user_id}:*"
            async for redis_key in self._redis.scan_iter(match=pattern, count=100):
                raw = await self._redis.get(redis_key)
                if raw is None:
                    continue
                sessions.append(ConversationSession.model_validate(json.loads(raw)))
                if len(sessions) >= limit:
                    break
        except Exception as exc:
            logger.warning("Conversation scan for user %s failed: %s", user_id, exc)
            return []
        for session in sessions:
            await self._index_session(session)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    async def delete_session(self, session_id: str, user_id: str | None = None) -> bool:
        """Remove a conversation. Returns False when it was already gone."""
        if self._redis is not None:
            removed = await self._redis.delete(f"{_REDIS_SESSION_PREFIX}{session_id}")
            if user_id:
                await self._redis.zrem(f"{_REDIS_USER_INDEX_PREFIX}{user_id}", session_id)
            return bool(removed)

        return self._memory_store.pop(session_id, None) is not None


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
