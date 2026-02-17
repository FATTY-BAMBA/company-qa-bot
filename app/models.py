"""
Database models for conversation logging and analytics.

Tables:
    - conversations: Groups messages into sessions
    - messages: Individual Q&A interactions with full metadata
"""

from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text,
    DateTime, Boolean, ForeignKey, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from app.config import DATABASE_URL

# ── Engine & Session ──
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for FastAPI — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════

class Conversation(Base):
    """Groups messages into a visitor session."""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_message_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    message_count = Column(Integer, default=0)

    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    """Individual Q&A interaction with full metadata."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    session_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Query & Response
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)

    # Retrieval metadata
    confidence = Column(Float, nullable=False)
    matches_found = Column(Integer, default=0)
    latency_seconds = Column(Float, default=0.0)
    model = Column(String(50), default="")

    # Sources (stored as JSON array)
    sources = Column(JSON, default=list)

    # Flags
    is_low_confidence = Column(Boolean, default=False, index=True)
    is_unanswered = Column(Boolean, default=False, index=True)

    conversation = relationship("Conversation", back_populates="messages")


# ── Create tables ──
def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
