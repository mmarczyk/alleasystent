from __future__ import annotations

"""
AllEasystent — AI assistant for e-store owners.

Entry point: FastAPI application with:
  - Facebook Messenger webhook
  - Allegro OAuth2 device flow endpoints
  - RAG indexing admin endpoints
  - Health check
  - Google OAuth2 login
"""

import asyncio
import base64
import hashlib
import hmac as _hmac
import logging
import os
import pathlib
import secrets as _secrets
import uuid as _uuid

# Disable ChromaDB telemetry before it is imported anywhere
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config.settings import get_settings
from webhooks.facebook_webhook import router as facebook_router

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ChromaDB has a bug where its posthog telemetry client is incompatible with
# the installed posthog version. Silence it — telemetry failures are harmless.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "Starting AllEasystent in %s mode on port %d",
        settings.app_env,
        settings.port,
    )
    # Ensure ChromaDB directory exists
    if settings.rag_backend == "chromadb":
        pathlib.Path(settings.chromadb_path).mkdir(parents=True, exist_ok=True)

    # Pre-warm the retriever in the background so sentence-transformers model
    # loading (can take 1-3 min on cold start) does NOT block server startup.
    # The server starts accepting requests immediately; first RAG query may
    # take a moment longer while the model finishes loading.
    async def _prewarm_retriever() -> None:
        try:
            from agents.rag.retriever import build_retriever
            await asyncio.to_thread(build_retriever)
            logger.info("ChromaDB retriever pre-warmed successfully")
        except Exception as exc:
            logger.warning("ChromaDB pre-warm failed (non-fatal): %s", exc)

    asyncio.create_task(_prewarm_retriever())

    # Backend order monitor: sends push notifications when new Allegro orders
    # arrive — works even when iOS PWA is backgrounded (JS can't run then).
    from services.order_monitor import run_order_monitor
    monitor_task = asyncio.create_task(run_order_monitor())

    yield

    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    logger.info("AllEasystent shutting down")


settings = get_settings()

# Unique token generated fresh on each process start — frontend uses it to
# detect backend redeployments and prompt the user to reload for new assets.
_SERVER_INSTANCE = _uuid.uuid4().hex[:12]

app = FastAPI(
    title="AllEasystent",
    description="AI assistant for e-store owners — Allegro + Facebook Messenger",
    version="0.1.0",
    lifespan=lifespan,
)

# Split deployment (GitHub Pages + Cloud Run): lock to specific origin so the
# browser accepts credentials=include requests.  All-in-one: wildcard is fine
# because frontend and backend share an origin and CORS never triggers.
def _origin_from_url(url: str) -> str:
    """Extract scheme+host from a full URL (strips path/query)."""
    from urllib.parse import urlparse
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"

_cors_origins = (
    [_origin_from_url(settings.frontend_url), "http://localhost:8080", "http://localhost:3000"]
    if settings.frontend_url
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=bool(settings.frontend_url),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Server-Instance"],
)


@app.middleware("http")
async def add_server_instance_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Server-Instance"] = _SERVER_INSTANCE
    return response

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(facebook_router)

# ── Module-level Orchestrator singleton (not re-created per request) ──────────
from agents.orchestrator import Orchestrator
_orchestrator = Orchestrator()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict:
    return {"status": "ok", "env": settings.app_env}


@app.get("/debug/redis", tags=["System"])
async def debug_redis(request: Request) -> dict:
    """Diagnose Redis connection and token storage."""
    redis_url = settings.redis_url
    if not redis_url:
        return {"redis_url_set": False, "note": "REDIS_URL env var is missing — set it in Railway"}

    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(redis_url, decode_responses=True)
        await r.ping()
        # Find all allegro token keys (one per user)
        keys = await r.keys("allegro:tokens:*")
        await r.aclose()
        # Check if current user has tokens
        current_user_key = None
        try:
            from services.auth_service import get_current_user
            user = await get_current_user(request)
            current_user_key = f"allegro:tokens:{user['sub']}"
        except Exception:
            pass
        return {
            "redis_url_set": True,
            "connected": True,
            "allegro_token_keys": keys,
            "current_user_has_tokens": current_user_key in keys if current_user_key else None,
        }
    except Exception as exc:
        return {"redis_url_set": True, "connected": False, "error": str(exc)}


# ── Auth (Allegro OAuth2 login) ───────────────────────────────────────────────
#
# Flow: /allegro/login → Allegro consent → /allegro/callback → JWT cookie → /

@app.get("/allegro/login", tags=["Auth"])
async def allegro_login():
    """Redirect browser to Allegro OAuth2 consent page."""
    if not settings.allegro_client_id:
        raise HTTPException(503, "Allegro credentials not configured")
    from urllib.parse import urlencode
    state = _secrets.token_urlsafe(32)
    params = urlencode({
        "response_type": "code",
        "client_id": settings.allegro_client_id,
        "redirect_uri": settings.allegro_redirect_uri,
        "prompt": "confirm",
        "state": state,
    })
    url = f"{settings.allegro_auth_url}/authorize?{params}"
    response = RedirectResponse(url=url, status_code=302)
    response.set_cookie("oauth_state", state, httponly=True, max_age=300, samesite="lax")
    return response


@app.get("/allegro/auth", tags=["Auth"])
async def allegro_auth():
    """Start Allegro OAuth2 flow (alias for /allegro/login — used by tests and older clients)."""
    if not settings.allegro_client_id:
        raise HTTPException(503, "Allegro credentials not configured")
    from urllib.parse import urlencode
    state = _secrets.token_urlsafe(32)
    params = urlencode({
        "response_type": "code",
        "client_id": settings.allegro_client_id,
        "redirect_uri": settings.allegro_redirect_uri,
        "prompt": "confirm",
        "state": state,
    })
    url = f"{settings.allegro_auth_url}/authorize?{params}"
    response = RedirectResponse(url=url, status_code=302)
    response.set_cookie("oauth_state", state, httponly=True, max_age=300, samesite="lax")
    return response


@app.get("/allegro/auth/status", tags=["Auth"])
async def allegro_auth_status(request: Request):
    """Return Allegro auth status for the current user. Does not require authentication."""
    from services.auth_service import get_current_user
    from services.allegro_service import AllegroService

    try:
        user = await get_current_user(request)
        user_id = user["sub"]
    except HTTPException:
        return {"status": "idle", "authenticated": False}

    service = AllegroService.get_instance(user_id)
    if service._tokens is None:
        await service._load_tokens_from_redis()

    if service._tokens is None:
        return {"status": "idle", "authenticated": False}

    return {"status": "authorized", "authenticated": True}


@app.get("/allegro/callback", tags=["Auth"])
async def allegro_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Handle Allegro OAuth2 callback, create session, redirect to app."""
    if error:
        raise HTTPException(400, f"Allegro OAuth error: {error}")
    stored_state = request.cookies.get("oauth_state")
    if not stored_state or stored_state != state:
        raise HTTPException(400, "Invalid OAuth state — please try logging in again")
    from services.allegro_service import AllegroService, exchange_allegro_code
    from services.auth_service import create_session_token
    try:
        login, tokens = await exchange_allegro_code(code)
    except Exception as exc:
        logger.error("Allegro code exchange failed: %s", exc)
        raise HTTPException(502, "Failed to exchange Allegro authorization code")
    # Persist tokens to Redis under the user's Allegro login
    service = AllegroService.get_instance(login)
    service._tokens = tokens
    await service._save_tokens()
    # Create JWT session
    session_token = create_session_token({"sub": login, "name": login})
    # Split deployment: redirect back to GitHub Pages after OAuth.
    # Cookie must be SameSite=None;Secure so the browser sends it on
    # cross-domain API calls from GitHub Pages to Cloud Run.
    is_split = bool(settings.frontend_url)
    response = RedirectResponse(url=settings.frontend_url or "/", status_code=302)
    response.set_cookie(
        "session", session_token,
        httponly=True,
        max_age=60 * 60 * 24 * 30,
        samesite="none" if is_split else "lax",
        secure=True if is_split else settings.is_production,
    )
    response.delete_cookie("oauth_state")
    logger.info("Allegro login successful for user: %s", login)
    return response


class AllegroExchangeRequest(BaseModel):
    code: str
    state: str


def _sign_oauth_state(nonce: str) -> str:
    """Return HMAC-SHA256 signed state token. Stateless — works across Cloud Run instances."""
    key = settings.jwt_secret.encode() or b"dev-fallback-key"
    sig = _hmac.new(key, nonce.encode(), hashlib.sha256).digest()
    return nonce + "." + base64.urlsafe_b64encode(sig).decode().rstrip("=")


def _verify_oauth_state(state: str) -> bool:
    nonce, _, sig_b64 = state.rpartition(".")
    if not nonce or not sig_b64:
        return False
    expected = _sign_oauth_state(nonce)
    return _hmac.compare_digest(expected, state)


def _oauth_redirect_uri() -> str:
    """Build the Allegro OAuth redirect URI.

    Use the frontend URL with a trailing slash so Allegro redirects to
    https://…/alleasystent/?code=…  GitHub Pages serves index.html directly
    for that URL (no 301 redirect), so ?code= is never dropped.
    Register exactly this URL (with trailing slash) in the Allegro portal.
    """
    if settings.frontend_url:
        base = settings.frontend_url.rstrip("/")
        return f"{base}/"
    return settings.allegro_redirect_uri


@app.get("/allegro/auth-url", tags=["Auth"])
async def allegro_auth_url():
    """Return Allegro OAuth URL for the frontend-initiated flow.
    Frontend redirects the user there; Allegro redirects back to oauth-callback.html with ?code=."""
    if not settings.allegro_client_id:
        raise HTTPException(503, "Allegro credentials not configured")
    from urllib.parse import urlencode
    nonce = _secrets.token_urlsafe(16)
    state = _sign_oauth_state(nonce)
    redirect_uri = _oauth_redirect_uri()
    params = urlencode({
        "response_type": "code",
        "client_id": settings.allegro_client_id,
        "redirect_uri": redirect_uri,
        "prompt": "confirm",
        "state": state,
    })
    return {"auth_url": f"{settings.allegro_auth_url}/authorize?{params}", "state": state}


@app.post("/allegro/exchange", tags=["Auth"])
async def allegro_exchange(body: AllegroExchangeRequest):
    """Exchange Allegro authorization code for token (frontend-initiated OAuth flow).
    Sets session cookie and returns user info as JSON."""
    if not _verify_oauth_state(body.state):
        raise HTTPException(400, "Invalid or expired OAuth state")
    from services.allegro_service import AllegroService, exchange_allegro_code
    from services.auth_service import create_session_token
    redirect_uri = _oauth_redirect_uri()
    try:
        login, tokens = await exchange_allegro_code(body.code, redirect_uri=redirect_uri)
    except Exception as exc:
        logger.error("Allegro code exchange failed: %s", exc)
        raise HTTPException(502, "Failed to exchange Allegro authorization code")
    service = AllegroService.get_instance(login)
    service._tokens = tokens
    await service._save_tokens()
    session_token = create_session_token({"sub": login, "name": login})
    # Return the token in the body so the frontend can store it in localStorage.
    # Safari ITP blocks cross-site Set-Cookie responses, so the cookie alone is
    # insufficient for split deployments (GitHub Pages → Cloud Run).
    response = JSONResponse({"ok": True, "name": login, "token": session_token})
    response.set_cookie(
        "session", session_token,
        httponly=True,
        max_age=60 * 60 * 24 * 30,
        samesite="none",
        secure=True,
    )
    return response


@app.get("/auth/logout", tags=["Auth"])
async def auth_logout():
    response = RedirectResponse(url=settings.frontend_url or "/", status_code=302)
    response.delete_cookie("session")
    return response


@app.get("/auth/me", tags=["Auth"])
async def auth_me(request: Request):
    from services.auth_service import get_current_user
    from services.allegro_service import AllegroService
    user = await get_current_user(request)
    # Allegro token may be missing after container restart (file-based storage lost).
    # Force re-login so the user completes OAuth again and gets a fresh token.
    svc = AllegroService.get_instance(user["sub"])
    if svc._tokens is None:
        await svc._load_tokens_from_redis()
    if svc._tokens is None:
        raise HTTPException(status_code=401, detail="allegro_auth_required")
    return {"sub": user["sub"], "name": user["name"]}


# ── RAG Admin ─────────────────────────────────────────────────────────────────

class IndexFileRequest(BaseModel):
    path: str
    file_type: str = "text"  # "text" | "json_catalog"
    metadata: dict | None = None


@app.post("/admin/rag/index-file", tags=["RAG Admin"])
async def index_file(body: IndexFileRequest) -> dict:
    """
    Index a local file into the vector store.
    Supported types: 'text' (plain text / Markdown), 'json_catalog' (product list JSON).
    """
    from agents.rag.indexer import DocumentIndexer

    indexer = DocumentIndexer()
    if body.file_type == "json_catalog":
        count = await indexer.index_json_catalog(body.path)
    else:
        count = await indexer.index_text_file(body.path, body.metadata)
    return {"indexed_chunks": count, "path": body.path}


class IndexFAQRequest(BaseModel):
    items: list[dict]  # [{"question": ..., "answer": ...}]


@app.post("/admin/rag/index-faq", tags=["RAG Admin"])
async def index_faq(body: IndexFAQRequest) -> dict:
    """Index FAQ items into the vector store."""
    from agents.rag.indexer import DocumentIndexer

    indexer = DocumentIndexer()
    count = await indexer.index_faq(body.items)
    return {"indexed_items": count}


@app.post("/admin/rag/index-allegro-offers", tags=["RAG Admin"])
async def index_allegro_offers() -> dict:
    """
    Fetch all active Allegro offers and index them into the RAG knowledge base.
    Run this periodically (e.g. daily) to keep the product knowledge up to date.
    """
    from agents.rag.indexer import DocumentIndexer
    from services.allegro_service import AllegroService

    allegro = AllegroService.get_instance()
    indexer = DocumentIndexer()
    offers, _ = await allegro.get_offers(limit=50)
    count = await indexer.index_allegro_offers(offers)
    return {"indexed_offers": count}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/admin/rag/query", tags=["RAG Admin"])
async def rag_query(body: QueryRequest) -> dict:
    """Test the RAG retriever directly."""
    from agents.rag.rag_agent import RAGAgent

    agent = RAGAgent()
    context, sources = await agent.retrieve(body.query)
    return {"context": context, "sources": sources}


# ── Direct query endpoint (for testing / API clients) ─────────────────────────

class DirectQueryRequest(BaseModel):
    message: str
    session_id: str = "api_session"
    sender_id: str = "api_user"


@app.post("/query", tags=["Chat"])
async def query(request_body: DirectQueryRequest, request: Request) -> dict:
    """
    Send a message directly to the orchestrator (bypassing Messenger).
    Useful for testing, admin dashboards, or other client integrations.
    Authentication via session cookie is optional; falls back to sender_id.
    """
    from models.conversation import ChannelType, IncomingMessage
    from services.auth_service import get_current_user

    try:
        user = await get_current_user(request)
        user_sub = user["sub"]
    except HTTPException:
        user_sub = request_body.sender_id

    message = IncomingMessage(
        text=request_body.message,
        session_id=f"{user_sub}:{request_body.session_id}",
        channel=ChannelType.API,
        sender_id=user_sub,
    )
    try:
        response = await _orchestrator.handle(message, user_id=user_sub)
    except Exception as exc:
        logger.exception("Orchestrator error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    # Fire-and-forget: log query for analytics (never blocks the response)
    asyncio.create_task(
        __import__("services.analytics_service", fromlist=["log_query"]).log_query(
            user_id=user_sub,
            text=request_body.message,
            intent=response.agent_type,
            response_len=len(response.text),
        )
    )

    return {
        "response": response.text,
        "agent": response.agent_type,
        "sources": response.sources,
    }


# ── Analytics Admin endpoints ─────────────────────────────────────────────────

@app.get("/admin/analytics", tags=["Analytics Admin"])
async def admin_analytics(request: Request) -> dict:
    """Return aggregated query analytics (intent counts, recent queries, gap suggestions)."""
    from services.auth_service import get_current_user
    from services.analytics_service import get_stats
    await get_current_user(request)  # auth required
    return await get_stats()


@app.post("/admin/analytics/analyze", tags=["Analytics Admin"])
async def admin_analytics_analyze(request: Request) -> dict:
    """Run LLM clustering on recent queries to surface missing features."""
    from services.auth_service import get_current_user
    from services.analytics_service import analyze_with_llm
    await get_current_user(request)  # auth required
    return await analyze_with_llm(
        client=_orchestrator._client,
        model_pool=_orchestrator._settings.model_pool(),
    )


@app.get("/allegro/order-event-stats", tags=["Allegro"])
async def allegro_order_event_stats(request: Request):
    """Return the latest Allegro event ID — used to establish monitoring baseline."""
    from services.auth_service import get_current_user
    from services.allegro_service import AllegroService, AllegroAuthError, AllegroAPIError

    user = await get_current_user(request)
    service = AllegroService.get_instance(user["sub"])
    if service._tokens is None:
        await service._load_tokens_from_redis()
    if service._tokens is None:
        raise HTTPException(401, "Not authenticated with Allegro")
    try:
        return await service.get_order_event_stats()
    except AllegroAuthError:
        raise HTTPException(401, "Allegro auth error")
    except AllegroAPIError as exc:
        raise HTTPException(502, str(exc))


@app.get("/allegro/orders/{order_id}", tags=["Allegro"])
async def allegro_get_order(order_id: str, request: Request):
    """Fetch details of a single Allegro order by checkout-form ID."""
    from services.auth_service import get_current_user
    from services.allegro_service import AllegroService, AllegroAuthError, AllegroAPIError

    user = await get_current_user(request)
    service = AllegroService.get_instance(user["sub"])
    if service._tokens is None:
        await service._load_tokens_from_redis()
    if service._tokens is None:
        raise HTTPException(401, "Not authenticated with Allegro")
    try:
        order = await service.get_order(order_id)
    except AllegroAuthError:
        raise HTTPException(401, "Allegro auth error")
    except AllegroAPIError as exc:
        raise HTTPException(502, str(exc))
    delivery = order.delivery or {}
    method_name = (delivery.get("method") or {}).get("name") or "—"
    return {
        "order_id": order.order_id,
        "buyer_login": order.buyer_login,
        "total_price": order.total_price,
        "currency": order.currency,
        "fulfillment_status": order.fulfillment_status,
        "delivery_method": method_name,
        "items": [{"name": li.offer_name, "quantity": li.quantity, "price": li.price} for li in order.line_items],
    }


@app.get("/allegro/order-events", tags=["Allegro"])
async def allegro_order_events(request: Request, since: str | None = None):
    """Poll Allegro order events for new READY_FOR_PROCESSING orders since a given event ID."""
    from services.auth_service import get_current_user
    from services.allegro_service import AllegroService, AllegroAuthError, AllegroAPIError

    user = await get_current_user(request)
    service = AllegroService.get_instance(user["sub"])
    if service._tokens is None:
        await service._load_tokens_from_redis()
    if service._tokens is None:
        raise HTTPException(401, "Not authenticated with Allegro")
    try:
        result = await service.get_order_events(since_event_id=since)
    except AllegroAuthError:
        raise HTTPException(401, "Allegro auth error")
    except AllegroAPIError as exc:
        raise HTTPException(502, str(exc))
    return result



@app.get("/allegro/pending-invoices", tags=["Allegro"])
async def allegro_pending_invoices(request: Request):
    """Return paid orders from the current month that require a VAT invoice but haven't received one."""
    from datetime import date
    from services.auth_service import get_current_user
    from services.allegro_service import AllegroService, AllegroAuthError, AllegroAPIError

    user = await get_current_user(request)
    service = AllegroService.get_instance(user["sub"])
    if service._tokens is None:
        await service._load_tokens_from_redis()
    if service._tokens is None:
        raise HTTPException(401, "Not authenticated with Allegro")
    try:
        today = date.today()
        orders = await service.get_orders_needing_invoice(month=today.month, year=today.year)
    except AllegroAuthError:
        raise HTTPException(401, "Allegro auth error")
    except AllegroAPIError as exc:
        raise HTTPException(502, str(exc))
    return {
        "orders": [
            {"order_id": o.order_id, "buyer": o.buyer_login, "total": o.total_price}
            for o in orders
        ],
        "count": len(orders),
    }


# ── Web Push ─────────────────────────────────────────────────────────────────

@app.get("/push/status", tags=["Push"])
async def push_status(request: Request):
    """Debug: show push configuration and subscriptions for the current user."""
    from services.auth_service import get_current_user
    from services.push_service import _get_subscriptions
    user = await get_current_user(request)
    subs = await _get_subscriptions(user["sub"])
    return {
        "user": user["sub"],
        "vapid_public_key_set": bool(settings.vapid_public_key),
        "vapid_private_key_set": bool(settings.vapid_private_key),
        "subscriptions_count": len(subs),
        "subscription_endpoints": [s.get("endpoint", "")[:60] + "…" for s in subs],
    }


@app.get("/push/vapid-public-key", tags=["Push"])
async def push_vapid_key():
    """Return the VAPID public key so the browser can subscribe to push."""
    if not settings.vapid_public_key:
        raise HTTPException(503, "Push notifications not configured — set VAPID_PUBLIC_KEY")
    return {"publicKey": settings.vapid_public_key}


@app.post("/push/subscribe", tags=["Push"])
async def push_subscribe(request: Request):
    """Store a browser push subscription for the current user."""
    from services.auth_service import get_current_user
    from services.push_service import save_subscription
    user = await get_current_user(request)
    body = await request.json()
    await save_subscription(user["sub"], body)
    return {"status": "subscribed"}


@app.delete("/push/subscribe", tags=["Push"])
async def push_unsubscribe(request: Request):
    """Remove a push subscription (user revoked permission or unsubscribed)."""
    from services.auth_service import get_current_user
    from services.push_service import remove_subscription
    user = await get_current_user(request)
    body = await request.json()
    await remove_subscription(user["sub"], body.get("endpoint", ""))
    return {"status": "unsubscribed"}


@app.post("/push/notify", tags=["Push"])
async def push_notify(request: Request):
    """Send a Web Push notification to all devices of the current user.

    Called by the client-side monitors when they detect new orders/invoices.
    The backend fans out the push to every subscribed device for this user,
    so the notification reaches iOS PWA, Android, and other desktop tabs.

    Optional body field `chatMessage`: if present, the formatted chat text is stored
    in Redis so that any device opening the app (including ones not running polling)
    can retrieve and display it via GET /push/pending.
    """
    from services.auth_service import get_current_user
    from services.push_service import send_push, store_pending_chat
    user = await get_current_user(request)
    body = await request.json()
    chat_message = body.get("chatMessage")
    if chat_message:
        await store_pending_chat(user["sub"], chat_message)
    await send_push(
        user_id=user["sub"],
        title=body.get("title", "AllEasystent"),
        body=body.get("body", ""),
        url=body.get("url", "/"),
    )
    return {"status": "sent"}


@app.get("/push/pending", tags=["Push"])
async def push_pending(request: Request):
    """Return (and remove) the oldest pending chat message for the current user.

    Called on app startup — lets devices that were offline during polling still
    receive the formatted order/invoice notification in their chat.
    Returns {chatMessage: string} or {chatMessage: null} when queue is empty.
    """
    from services.auth_service import get_current_user
    from services.push_service import pop_pending_chat
    user = await get_current_user(request)
    text = await pop_pending_chat(user["sub"])
    return {"chatMessage": text}


# ── Static UI ─────────────────────────────────────────────────────────────────
# Serve web/ at root — must be mounted AFTER all API routes so they take priority.
_web_dir = pathlib.Path(__file__).parent / "web"
if _web_dir.exists():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
