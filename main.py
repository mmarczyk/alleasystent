from __future__ import annotations

"""
AllEasystent — AI assistant for e-store owners.

Entry point: FastAPI application with:
  - Facebook Messenger webhook
  - Allegro OAuth2 device flow endpoints
  - RAG indexing admin endpoints
  - Health check
"""

import logging
import os

# Disable ChromaDB telemetry before it is imported anywhere
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
        import pathlib
        pathlib.Path(settings.chromadb_path).mkdir(parents=True, exist_ok=True)
    yield
    logger.info("AllEasystent shutting down")


settings = get_settings()
app = FastAPI(
    title="AllEasystent",
    description="AI assistant for e-store owners — Allegro + Facebook Messenger",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(facebook_router)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict:
    return {"status": "ok", "env": settings.app_env}


# ── Allegro OAuth2 device flow ────────────────────────────────────────────────
#
# New UX: owner opens /allegro/auth in the browser → gets redirected straight
# to Allegro's consent page → approves → server detects it in the background.
# No manual polling or copy-pasting codes required.

import asyncio as _asyncio

# Shared auth state (in-memory; single-process is fine for one-owner setups)
_allegro_auth_state: dict = {"status": "idle", "error": None}


async def _background_poll(device_code: str, interval: int) -> None:
    """Runs as an asyncio Task — polls Allegro until the owner approves or it times out."""
    from services.allegro_service import AllegroAuthError, AllegroService

    _allegro_auth_state["status"] = "pending"
    _allegro_auth_state["error"] = None
    service = AllegroService()
    try:
        success = await service.poll_device_flow(device_code, interval=interval)
        _allegro_auth_state["status"] = "authorized" if success else "expired"
    except AllegroAuthError as exc:
        _allegro_auth_state["status"] = "error"
        _allegro_auth_state["error"] = str(exc)
        logger.error("Allegro background auth failed: %s", exc)


@app.get("/allegro/auth", tags=["Allegro"])
async def allegro_start_auth():
    """
    Start Allegro authorization.

    Redirects the browser directly to Allegro's consent page (verification_uri_complete).
    The server polls in the background; check /allegro/auth/status for the result.
    """
    from fastapi.responses import RedirectResponse
    from services.allegro_service import AllegroService

    if not settings.allegro_client_id:
        raise HTTPException(status_code=503, detail="Allegro credentials not configured")

    service = AllegroService()
    flow = await service.start_device_flow()

    device_code = flow.get("device_code", "")
    interval = int(flow.get("interval", 5))
    verification_uri = flow.get("verification_uri_complete") or flow.get("verification_uri", "")

    if not verification_uri:
        raise HTTPException(status_code=502, detail=f"Allegro did not return a verification URL. Response: {flow}")

    # Kick off background polling so the server catches the approval automatically
    _asyncio.create_task(_background_poll(device_code, interval))
    logger.info("Allegro device flow started — redirecting to %s", verification_uri)

    return RedirectResponse(url=verification_uri, status_code=302)


@app.get("/allegro/auth/status", tags=["Allegro"])
async def allegro_auth_status() -> dict:
    """
    Check whether the background authorization has completed.

    Statuses:
      idle       — no auth started yet
      pending    — waiting for owner to approve on Allegro
      authorized — tokens saved, API calls will work
      expired    — owner didn't approve in time; restart /allegro/auth
      error      — something went wrong; see 'error' field
    """
    from services.allegro_service import AllegroService

    state = dict(_allegro_auth_state)
    if state["status"] == "authorized":
        # Double-check tokens are actually on disk / in memory
        service = AllegroService()
        state["authenticated"] = service._tokens is not None
    return state


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

    allegro = AllegroService()
    indexer = DocumentIndexer()
    offers = await allegro.get_offers(limit=50)
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


@app.post("/query", tags=["Query"])
async def direct_query(body: DirectQueryRequest) -> dict:
    """
    Send a message directly to the orchestrator (bypassing Messenger).
    Useful for testing, admin dashboards, or other client integrations.
    """
    from agents.orchestrator import Orchestrator
    from models.conversation import ChannelType, IncomingMessage

    orchestrator = Orchestrator()
    message = IncomingMessage(
        channel=ChannelType.API,
        sender_id=body.sender_id,
        session_id=body.session_id,
        text=body.message,
    )
    try:
        response = await orchestrator.handle(message)
    except Exception as exc:
        logger.exception("Orchestrator error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")
    return {
        "response": response.text,
        "agent": response.agent_type,
        "sources": response.sources,
    }


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
