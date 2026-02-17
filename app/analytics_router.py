"""
Analytics API Router.

Separate router for management-facing analytics endpoints.
All endpoints require API key authentication.

Endpoints:
    GET /api/analytics/top-questions
    GET /api/analytics/unanswered
    GET /api/analytics/trends
    GET /api/analytics/categories
    GET /api/analytics/low-confidence
    GET /api/analytics/weekly-summary
    GET /api/analytics/overview
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends, Security
from fastapi.security import APIKeyHeader

from app.config import ANALYTICS_API_KEY
from app.analytics import (
    get_top_questions,
    get_unanswered_questions,
    get_engagement_trends,
    generate_weekly_summary,
    get_category_breakdown,
    get_low_confidence_responses,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# ── Auth ──
api_key_header = APIKeyHeader(name="X-Analytics-Key", auto_error=False)


async def verify_analytics_key(api_key: str = Security(api_key_header)):
    """Verify the analytics API key."""
    if not ANALYTICS_API_KEY:
        return True  # No key configured = open access (dev mode)
    if api_key != ANALYTICS_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid analytics API key")
    return True


# ═══════════════════════════════════════════════════════════
# 1. Most Asked Questions
# ═══════════════════════════════════════════════════════════

@router.get("/top-questions")
async def top_questions(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=100),
    _auth: bool = Depends(verify_analytics_key),
):
    """Top questions ranked by frequency."""
    return get_top_questions(days=days, limit=limit)


# ═══════════════════════════════════════════════════════════
# 2. Unanswered Questions (Knowledge Gaps)
# ═══════════════════════════════════════════════════════════

@router.get("/unanswered")
async def unanswered(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=100),
    _auth: bool = Depends(verify_analytics_key),
):
    """Questions the bot could not answer — knowledge base gaps."""
    return get_unanswered_questions(days=days, limit=limit)


# ═══════════════════════════════════════════════════════════
# 3. Engagement Trends
# ═══════════════════════════════════════════════════════════

@router.get("/trends")
async def trends(
    days: int = Query(default=30, ge=1, le=365),
    _auth: bool = Depends(verify_analytics_key),
):
    """Daily message counts, unique sessions, and performance metrics."""
    return get_engagement_trends(days=days)


# ═══════════════════════════════════════════════════════════
# 4. AI-Generated Weekly Summary
# ═══════════════════════════════════════════════════════════

@router.get("/weekly-summary")
async def weekly_summary(
    _auth: bool = Depends(verify_analytics_key),
):
    """AI-generated management summary of the past 7 days."""
    return generate_weekly_summary()


# ═══════════════════════════════════════════════════════════
# 5. Category Breakdown
# ═══════════════════════════════════════════════════════════

@router.get("/categories")
async def categories(
    days: int = Query(default=30, ge=1, le=365),
    _auth: bool = Depends(verify_analytics_key),
):
    """Query volume broken down by topic category."""
    return get_category_breakdown(days=days)


# ═══════════════════════════════════════════════════════════
# 6. Low Confidence Responses
# ═══════════════════════════════════════════════════════════

@router.get("/low-confidence")
async def low_confidence(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=100),
    _auth: bool = Depends(verify_analytics_key),
):
    """Responses where the bot had low confidence — needs KB improvement."""
    return get_low_confidence_responses(days=days, limit=limit)


# ═══════════════════════════════════════════════════════════
# Overview (combines key metrics)
# ═══════════════════════════════════════════════════════════

@router.get("/overview")
async def overview(
    days: int = Query(default=30, ge=1, le=365),
    _auth: bool = Depends(verify_analytics_key),
):
    """Combined overview of all key analytics metrics."""
    top = get_top_questions(days=days, limit=5)
    gaps = get_unanswered_questions(days=days, limit=5)
    engagement = get_engagement_trends(days=days)
    cats = get_category_breakdown(days=days)
    low = get_low_confidence_responses(days=days, limit=5)

    return {
        "period_days": days,
        "engagement": {
            "total_messages": engagement["total_messages"],
            "total_sessions": engagement["total_sessions"],
            "avg_messages_per_day": engagement["avg_messages_per_day"],
        },
        "top_questions": top["top_questions"][:5],
        "unanswered": {
            "count": gaps["total_unanswered"],
            "rate_percent": gaps["unanswered_rate_percent"],
            "recent": gaps["unanswered_questions"][:5],
        },
        "categories": cats["categories"][:10],
        "low_confidence": {
            "count": low["total_low_confidence"],
            "rate_percent": low["low_confidence_rate_percent"],
            "worst": low["responses"][:5],
        },
    }
