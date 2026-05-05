from __future__ import annotations

"""GCP service layer: Firestore (conversation history) and Pub/Sub (async messaging)."""

import json
import logging
from datetime import datetime
from typing import Any

from config.settings import get_settings
from models.conversation import ChannelType, ConversationSession, MessageRole

logger = logging.getLogger(__name__)


class FirestoreService:
    """
    Manages conversation sessions in Firestore.

    Falls back to in-memory store when running locally without GCP credentials,
    so the system works in development without any GCP setup.
    """

    def __init__(self):
        self._settings = get_settings()
        self._db = None
        self._memory_store: dict[str, dict] = {}  # local fallback
        self._init_firestore()

    def _init_firestore(self) -> None:
        if not self._settings.gcp_project_id:
            logger.info("GCP project not configured — using in-memory store")
            return
        try:
            from google.cloud import firestore

            self._db = firestore.AsyncClient(project=self._settings.gcp_project_id)
            logger.info("Firestore client initialized")
        except ImportError:
            logger.warning("google-cloud-firestore not installed — using in-memory store")
        except Exception as exc:
            logger.warning("Firestore init failed (%s) — using in-memory store", exc)

    async def get_session(self, session_id: str) -> ConversationSession | None:
        if self._db is None:
            data = self._memory_store.get(session_id)
            if data is None:
                return None
            return ConversationSession.model_validate(data)

        doc = await self._db.collection(
            self._settings.firestore_collection_conversations
        ).document(session_id).get()
        if not doc.exists:
            return None
        return ConversationSession.model_validate(doc.to_dict())

    async def save_session(self, session: ConversationSession) -> None:
        session.updated_at = datetime.utcnow()
        data = json.loads(session.model_dump_json())

        if self._db is None:
            self._memory_store[session.session_id] = data
            return

        await self._db.collection(
            self._settings.firestore_collection_conversations
        ).document(session.session_id).set(data)

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
        if self._db is None:
            sessions = [
                ConversationSession.model_validate(v) for v in self._memory_store.values()
            ]
            if channel:
                sessions = [s for s in sessions if s.channel == channel]
            return sessions[:limit]

        query = self._db.collection(self._settings.firestore_collection_conversations)
        if channel:
            query = query.where("channel", "==", channel.value)
        query = query.limit(limit)
        docs = await query.get()
        return [ConversationSession.model_validate(doc.to_dict()) for doc in docs]


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
