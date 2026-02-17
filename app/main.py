"""
Company Q&A Bot — FastAPI Application.

Endpoints:
    POST /api/webhooks/sheets-update    — Google Sheets edit webhook
    POST /api/chat                      — Visitor chat query
    POST /api/admin/reindex             — Manual re-index trigger
    GET  /api/health                    — Health check
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    SHEETS_WEBHOOK_SECRET,
    COMPANY_QA_SPREADSHEET_ID,
    COMPANY_QA_SHEET_NAME,
    APP_ENV,
    LOG_LEVEL,
)
from app.indexer import reindex_company_qa
from app.chat import chat, chat_stream, clear_cache
from app.logger import log_interaction
from app.models import init_db
from app.analytics_router import router as analytics_router

# ── Logging ──
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ──
app = FastAPI(
    title="Company Q&A Bot",
    description="AI-powered Q&A bot with Google Sheets knowledge base",
    version="1.0.0",
)

# ── CORS ──
# Update origins for production deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if APP_ENV == "development" else [
        "https://your-company-website.com",  # TODO: Replace with actual domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include analytics router ──
app.include_router(analytics_router)

# ── Initialize database on startup ──
@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("Database initialized")


# ═══════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════

class WebhookPayload(BaseModel):
    spreadsheet_id: str
    sheet_name: str = "Sheet1"
    timestamp: str = ""
    secret: str


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float
    session_id: str
    timestamp: str
    latency_seconds: float
    matches_found: int


# ═══════════════════════════════════════════════════════════
# Webhook Endpoint (Step 3 — receives Google Apps Script POST)
# ═══════════════════════════════════════════════════════════

@app.post("/api/webhooks/sheets-update")
async def handle_sheets_webhook(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
):
    """
    Receives webhook from Google Apps Script when the sheet is edited.
    Validates the shared secret, then triggers re-indexing in the background.
    Returns 200 immediately so the webhook doesn't time out.
    """
    # Validate secret
    if payload.secret != SHEETS_WEBHOOK_SECRET:
        logger.warning("Webhook received with invalid secret")
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    logger.info(
        f"Sheet update webhook received: "
        f"{payload.spreadsheet_id} / {payload.sheet_name}"
    )

    # Trigger re-indexing in background
    background_tasks.add_task(
        reindex_company_qa,
        spreadsheet_id=payload.spreadsheet_id,
        sheet_name=payload.sheet_name,
    )

    return {
        "status": "accepted",
        "message": "Re-indexing triggered",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# Chat Endpoint (visitor queries)
# ═══════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Process a visitor's question against the Q&A knowledge base.

    Embeds the query, retrieves relevant content from Pinecone,
    and generates a grounded response via the LLM.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"Chat query [{session_id}]: {request.query[:80]}...")

    try:
        result = chat(
            query=request.query,
            conversation_history=request.conversation_history,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate response")

    # Log the interaction for analytics
    try:
        log_interaction(
            session_id=session_id,
            query=request.query,
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
            matches_found=result["matches_found"],
            latency_seconds=result["latency_seconds"],
            model=result["model"],
        )
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}", exc_info=True)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        session_id=session_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_seconds=result["latency_seconds"],
        matches_found=result["matches_found"],
    )


# ═══════════════════════════════════════════════════════════
# Streaming Chat Endpoint (SSE)
# ═══════════════════════════════════════════════════════════

@app.post("/api/chat/stream")
async def handle_chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events.
    The visitor sees text appearing in real-time instead of
    waiting for the full response.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Stream chat [{session_id}]: {request.query[:80]}...")

    def event_generator():
        full_answer = ""
        metadata = {}

        for event in chat_stream(
            query=request.query,
            conversation_history=request.conversation_history,
        ):
            yield event

            # Parse SSE events to capture metadata and full answer
            if event.startswith("data: "):
                try:
                    import json as _json
                    payload = _json.loads(event[6:].strip())
                    if payload.get("type") == "metadata":
                        metadata = payload
                    elif payload.get("type") == "chunk":
                        full_answer += payload.get("text", "")
                except Exception:
                    pass

        # Log after streaming completes
        try:
            log_interaction(
                session_id=session_id,
                query=request.query,
                answer=full_answer,
                confidence=metadata.get("confidence", 0.0),
                sources=metadata.get("sources", []),
                matches_found=metadata.get("matches_found", 0),
                latency_seconds=metadata.get("latency_seconds", 0.0),
                model=metadata.get("model", ""),
            )
        except Exception as e:
            logger.error(f"Failed to log streamed interaction: {e}", exc_info=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )


# ═══════════════════════════════════════════════════════════
# Admin Endpoint (manual re-index)
# ═══════════════════════════════════════════════════════════

@app.post("/api/admin/reindex")
async def manual_reindex(background_tasks: BackgroundTasks):
    """
    Manually trigger a full re-index of the knowledge base.
    Useful for initial setup, debugging, or forcing a refresh.
    """
    logger.info("Manual re-index triggered via admin endpoint")

    background_tasks.add_task(
        reindex_company_qa,
        spreadsheet_id=COMPANY_QA_SPREADSHEET_ID,
        sheet_name=COMPANY_QA_SHEET_NAME,
    )

    # Clear query cache since knowledge base is changing
    clear_cache()

    return {
        "status": "accepted",
        "message": "Re-indexing triggered",
        "spreadsheet_id": COMPANY_QA_SPREADSHEET_ID,
        "sheet_name": COMPANY_QA_SHEET_NAME,
    }


# ═══════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "company-qa-bot",
        "environment": APP_ENV,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# Static Files & Demo Pages
# ═══════════════════════════════════════════════════════════

import os
from fastapi.responses import FileResponse, RedirectResponse

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

@app.get("/")
async def root():
    """Redirect root to chat demo."""
    return RedirectResponse(url="/chat")

@app.get("/chat")
async def chat_page():
    """Serve the chat demo page."""
    return FileResponse(os.path.join(static_dir, "chat.html"))

@app.get("/dashboard")
async def dashboard_page():
    """Serve the analytics dashboard."""
    return FileResponse(os.path.join(static_dir, "dashboard.html"))