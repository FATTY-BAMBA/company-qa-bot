"""
Analytics Engine.

Provides six analytics features:
1. Most asked questions (ranked)
2. Unanswered questions (knowledge gaps)
3. Visitor engagement trends (daily/weekly)
4. AI-generated weekly summary for management
5. Category breakdown
6. Low confidence responses
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from collections import Counter

from sqlalchemy import func, desc, cast, Date
from openai import OpenAI

from app.models import SessionLocal, Message, Conversation
from app.config import OPENAI_API_KEY, LOW_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def _get_date_filter(days: int = 30):
    """Return a datetime for filtering by recent period."""
    return datetime.now(timezone.utc) - timedelta(days=days)


# ═══════════════════════════════════════════════════════════
# 1. Most Asked Questions (Ranked)
# ═══════════════════════════════════════════════════════════

def get_top_questions(days: int = 30, limit: int = 20) -> dict:
    """
    Returns the most frequently asked questions, ranked by count.
    Groups similar questions together using exact match for now.
    """
    db = SessionLocal()
    try:
        since = _get_date_filter(days)
        results = (
            db.query(
                Message.query,
                func.count(Message.id).label("count"),
                func.avg(Message.confidence).label("avg_confidence"),
            )
            .filter(Message.timestamp >= since)
            .group_by(Message.query)
            .order_by(desc("count"))
            .limit(limit)
            .all()
        )

        questions = []
        for row in results:
            questions.append({
                "question": row[0],
                "count": row[1],
                "avg_confidence": round(row[2], 4) if row[2] else 0,
            })

        total_messages = (
            db.query(func.count(Message.id))
            .filter(Message.timestamp >= since)
            .scalar()
        )

        return {
            "period_days": days,
            "total_messages": total_messages,
            "top_questions": questions,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# 2. Unanswered Questions (Knowledge Gaps)
# ═══════════════════════════════════════════════════════════

def get_unanswered_questions(days: int = 30, limit: int = 20) -> dict:
    """
    Returns questions the bot could not answer (triggered fallback).
    These represent gaps in the knowledge base that the client should address.
    """
    db = SessionLocal()
    try:
        since = _get_date_filter(days)
        results = (
            db.query(
                Message.query,
                Message.confidence,
                Message.timestamp,
            )
            .filter(
                Message.timestamp >= since,
                Message.is_unanswered == True,
            )
            .order_by(desc(Message.timestamp))
            .limit(limit)
            .all()
        )

        questions = []
        for row in results:
            questions.append({
                "question": row[0],
                "confidence": round(row[1], 4),
                "asked_at": row[2].isoformat(),
            })

        total_unanswered = (
            db.query(func.count(Message.id))
            .filter(
                Message.timestamp >= since,
                Message.is_unanswered == True,
            )
            .scalar()
        )

        total_messages = (
            db.query(func.count(Message.id))
            .filter(Message.timestamp >= since)
            .scalar()
        )

        unanswered_rate = round(total_unanswered / total_messages * 100, 1) if total_messages > 0 else 0

        return {
            "period_days": days,
            "total_unanswered": total_unanswered,
            "total_messages": total_messages,
            "unanswered_rate_percent": unanswered_rate,
            "unanswered_questions": questions,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# 3. Visitor Engagement Trends (Daily/Weekly)
# ═══════════════════════════════════════════════════════════

def get_engagement_trends(days: int = 30) -> dict:
    """
    Returns daily message counts and unique session counts
    for trend visualization.
    """
    db = SessionLocal()
    try:
        since = _get_date_filter(days)

        # Daily message counts
        daily_messages = (
            db.query(
                func.date(Message.timestamp).label("date"),
                func.count(Message.id).label("message_count"),
                func.count(func.distinct(Message.session_id)).label("unique_sessions"),
                func.avg(Message.confidence).label("avg_confidence"),
                func.avg(Message.latency_seconds).label("avg_latency"),
            )
            .filter(Message.timestamp >= since)
            .group_by(func.date(Message.timestamp))
            .order_by(func.date(Message.timestamp))
            .all()
        )

        trends = []
        for row in daily_messages:
            trends.append({
                "date": row[0] if isinstance(row[0], str) else (row[0].isoformat() if row[0] else None),
                "message_count": row[1],
                "unique_sessions": row[2],
                "avg_confidence": round(row[3], 4) if row[3] else 0,
                "avg_latency": round(row[4], 3) if row[4] else 0,
            })

        # Summary stats
        total_messages = sum(t["message_count"] for t in trends)
        total_sessions = (
            db.query(func.count(func.distinct(Message.session_id)))
            .filter(Message.timestamp >= since)
            .scalar()
        )
        avg_messages_per_day = round(total_messages / max(len(trends), 1), 1)

        return {
            "period_days": days,
            "total_messages": total_messages,
            "total_sessions": total_sessions,
            "avg_messages_per_day": avg_messages_per_day,
            "daily_trends": trends,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# 4. AI-Generated Weekly Summary
# ═══════════════════════════════════════════════════════════

def generate_weekly_summary() -> dict:
    """
    Uses GPT to generate a management-friendly weekly summary
    based on conversation data from the past 7 days.
    """
    db = SessionLocal()
    try:
        since = _get_date_filter(7)

        # Gather stats
        total_messages = (
            db.query(func.count(Message.id))
            .filter(Message.timestamp >= since)
            .scalar()
        )

        total_sessions = (
            db.query(func.count(func.distinct(Message.session_id)))
            .filter(Message.timestamp >= since)
            .scalar()
        )

        unanswered_count = (
            db.query(func.count(Message.id))
            .filter(
                Message.timestamp >= since,
                Message.is_unanswered == True,
            )
            .scalar()
        )

        low_confidence_count = (
            db.query(func.count(Message.id))
            .filter(
                Message.timestamp >= since,
                Message.is_low_confidence == True,
            )
            .scalar()
        )

        avg_confidence = (
            db.query(func.avg(Message.confidence))
            .filter(Message.timestamp >= since)
            .scalar()
        )

        # Top questions
        top_questions = (
            db.query(Message.query, func.count(Message.id).label("cnt"))
            .filter(Message.timestamp >= since)
            .group_by(Message.query)
            .order_by(desc("cnt"))
            .limit(10)
            .all()
        )

        # Unanswered questions
        unanswered = (
            db.query(Message.query)
            .filter(
                Message.timestamp >= since,
                Message.is_unanswered == True,
            )
            .limit(10)
            .all()
        )

        # Build data summary for GPT
        data_summary = f"""
過去 7 天的客服聊天機器人數據摘要：

總訊息數：{total_messages}
總訪客數（獨立會話）：{total_sessions}
平均信心度：{round(avg_confidence, 2) if avg_confidence else 'N/A'}
無法回答的問題數：{unanswered_count}
低信心度回應數：{low_confidence_count}

最常被問到的問題（前10名）：
{chr(10).join([f'  {i+1}. "{q[0]}" ({q[1]} 次)' for i, q in enumerate(top_questions)])}

無法回答的問題：
{chr(10).join([f'  - "{q[0]}"' for q in unanswered]) if unanswered else '  （無）'}
"""

        # Generate summary via GPT
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一位數據分析師，負責為管理層撰寫每週客服聊天機器人報告。"
                        "請用繁體中文撰寫簡潔、有洞察力的摘要。"
                        "包含以下內容：1) 本週亮點 2) 訪客最關心的話題 "
                        "3) 知識庫缺口（無法回答的問題）4) 改善建議。"
                        "語氣專業但易讀，適合高階主管閱讀。"
                    ),
                },
                {"role": "user", "content": data_summary},
            ],
            temperature=0.4,
            max_tokens=1000,
        )

        summary_text = response.choices[0].message.content

        return {
            "period": "past_7_days",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "total_messages": total_messages,
                "total_sessions": total_sessions,
                "avg_confidence": round(avg_confidence, 4) if avg_confidence else 0,
                "unanswered_count": unanswered_count,
                "low_confidence_count": low_confidence_count,
            },
            "summary": summary_text,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# 5. Category Breakdown
# ═══════════════════════════════════════════════════════════

def get_category_breakdown(days: int = 30) -> dict:
    """
    Breaks down queries by the categories of their matched sources.
    Shows which topics visitors ask about most.
    """
    db = SessionLocal()
    try:
        since = _get_date_filter(days)
        messages = (
            db.query(Message.sources)
            .filter(Message.timestamp >= since)
            .all()
        )

        category_counts = Counter()
        for row in messages:
            sources = row[0] if row[0] else []
            for source in sources:
                category = source.get("category", "")
                if category:
                    category_counts[category] += 1

        # Sort by count
        categories = [
            {"category": cat, "count": count}
            for cat, count in category_counts.most_common()
        ]

        total = sum(c["count"] for c in categories)
        for cat in categories:
            cat["percentage"] = round(cat["count"] / total * 100, 1) if total > 0 else 0

        return {
            "period_days": days,
            "total_categorized_matches": total,
            "categories": categories,
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# 6. Low Confidence Responses
# ═══════════════════════════════════════════════════════════

def get_low_confidence_responses(days: int = 30, limit: int = 20) -> dict:
    """
    Returns responses where the bot had low confidence.
    These indicate areas where the knowledge base needs improvement.
    """
    db = SessionLocal()
    try:
        since = _get_date_filter(days)
        results = (
            db.query(
                Message.query,
                Message.answer,
                Message.confidence,
                Message.matches_found,
                Message.timestamp,
                Message.sources,
            )
            .filter(
                Message.timestamp >= since,
                Message.is_low_confidence == True,
            )
            .order_by(Message.confidence)
            .limit(limit)
            .all()
        )

        responses = []
        for row in results:
            responses.append({
                "question": row[0],
                "answer": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                "confidence": round(row[2], 4),
                "matches_found": row[3],
                "asked_at": row[4].isoformat(),
                "top_source": row[5][0] if row[5] else None,
            })

        total_low = (
            db.query(func.count(Message.id))
            .filter(
                Message.timestamp >= since,
                Message.is_low_confidence == True,
            )
            .scalar()
        )

        total_messages = (
            db.query(func.count(Message.id))
            .filter(Message.timestamp >= since)
            .scalar()
        )

        low_rate = round(total_low / total_messages * 100, 1) if total_messages > 0 else 0

        return {
            "period_days": days,
            "threshold": LOW_CONFIDENCE_THRESHOLD,
            "total_low_confidence": total_low,
            "total_messages": total_messages,
            "low_confidence_rate_percent": low_rate,
            "responses": responses,
        }
    finally:
        db.close()
