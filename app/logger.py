"""
Conversation Logger.

Stores every chat interaction to the database for analytics.
Handles session management (creating new sessions, updating existing ones).
"""

import logging
from datetime import datetime, timezone

from app.models import SessionLocal, Conversation, Message
from app.config import LOW_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Fallback message keywords to detect unanswered queries
FALLBACK_INDICATORS = [
    "無法回答",
    "聯繫我們",
    "聯繫客服",
    "support@",
    "02-1234-5678",
]


def is_fallback_response(answer: str) -> bool:
    """Check if the bot's answer is a fallback/unable-to-answer response."""
    return any(indicator in answer for indicator in FALLBACK_INDICATORS)


def log_interaction(
    session_id: str,
    query: str,
    answer: str,
    confidence: float,
    sources: list,
    matches_found: int,
    latency_seconds: float,
    model: str,
):
    """
    Log a single chat interaction to the database.

    Creates a new conversation if this session_id hasn't been seen,
    or appends to an existing conversation.
    """
    db = SessionLocal()
    try:
        # Find or create conversation
        conversation = (
            db.query(Conversation)
            .filter(Conversation.session_id == session_id)
            .first()
        )

        if not conversation:
            conversation = Conversation(
                session_id=session_id,
                started_at=datetime.now(timezone.utc),
                last_message_at=datetime.now(timezone.utc),
                message_count=0,
            )
            db.add(conversation)
            db.flush()

        # Update conversation
        conversation.last_message_at = datetime.now(timezone.utc)
        conversation.message_count += 1

        # Determine flags
        low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
        unanswered = is_fallback_response(answer)

        # Create message
        message = Message(
            conversation_id=conversation.id,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            query=query,
            answer=answer,
            confidence=confidence,
            matches_found=matches_found,
            latency_seconds=latency_seconds,
            model=model,
            sources=sources,
            is_low_confidence=low_confidence,
            is_unanswered=unanswered,
        )
        db.add(message)
        db.commit()

        logger.info(
            f"Logged interaction [{session_id}] — "
            f"confidence: {confidence}, "
            f"low_confidence: {low_confidence}, "
            f"unanswered: {unanswered}"
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log interaction: {e}", exc_info=True)
    finally:
        db.close()
