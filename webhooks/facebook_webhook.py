from __future__ import annotations

"""
Facebook Messenger webhook router.

Handles:
  GET /webhook/facebook  — webhook verification challenge
  POST /webhook/facebook — incoming message events
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from agents.communication.facebook_agent import FacebookCommunicationAgent
from agents.orchestrator import Orchestrator
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook/facebook", tags=["Facebook Webhook"])

# Shared singletons (initialized once at startup)
_communication_agent: FacebookCommunicationAgent | None = None
_orchestrator: Orchestrator | None = None


def get_communication_agent() -> FacebookCommunicationAgent:
    global _communication_agent
    if _communication_agent is None:
        _communication_agent = FacebookCommunicationAgent()
    return _communication_agent


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


@router.get("")
async def verify_webhook(
    hub_mode: str | None = None,
    hub_verify_token: str | None = None,
    hub_challenge: str | None = None,
) -> Response:
    """
    Facebook webhook verification.
    Facebook sends a GET with hub.verify_token; we echo hub.challenge if it matches.
    """
    settings = get_settings()
    if hub_mode == "subscribe" and hub_verify_token == settings.facebook_verify_token:
        logger.info("Facebook webhook verified successfully")
        return PlainTextResponse(content=hub_challenge or "")
    logger.warning("Webhook verification failed — token mismatch")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


@router.post("")
async def receive_message(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    Receive Facebook Messenger events.

    Facebook expects a 200 response within 5 seconds.
    We acknowledge immediately and process in the background.
    """
    settings = get_settings()

    # Validate signature
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if settings.facebook_app_secret:
        comm_agent = get_communication_agent()
        if not comm_agent._fb.verify_webhook_signature(raw_body, signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if payload.get("object") != "page":
        return {"status": "ignored"}

    background_tasks.add_task(_process_webhook, payload)
    return {"status": "ok"}


async def _process_webhook(payload: dict[str, Any]) -> None:
    """Background task: parse events → orchestrate → send responses."""
    comm_agent = get_communication_agent()
    orchestrator = get_orchestrator()

    try:
        messages = await comm_agent.parse_incoming(payload)
    except Exception as exc:
        logger.exception("Failed to parse Facebook payload: %s", exc)
        return

    for msg in messages:
        try:
            response = await orchestrator.handle(msg)
            await comm_agent.send_response(msg, response)
        except Exception as exc:
            logger.exception("Failed to process message from %s: %s", msg.sender_id, exc)
            # Best-effort error reply
            try:
                from services.facebook_service import FacebookService
                fb = FacebookService()
                await fb.send_text_message(
                    msg.sender_id,
                    "Sorry, something went wrong. Please try again in a moment.",
                )
            except Exception:
                pass
