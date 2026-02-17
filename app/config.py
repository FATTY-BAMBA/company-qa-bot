"""
Configuration module — loads all environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Google Sheets ──
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
GOOGLE_SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
COMPANY_QA_SPREADSHEET_ID = os.environ.get("COMPANY_QA_SPREADSHEET_ID", "")
COMPANY_QA_SHEET_NAME = os.environ.get("COMPANY_QA_SHEET_NAME", "Sheet1")

# ── Webhook ──
SHEETS_WEBHOOK_SECRET = os.environ.get("SHEETS_WEBHOOK_SECRET", "")

# ── Pinecone ──
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "")
COMPANY_QA_NAMESPACE = os.environ.get("COMPANY_QA_NAMESPACE", "company-qa-bot")

# ── OpenAI ──
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ── App ──
APP_ENV = os.environ.get("APP_ENV", "development")
APP_PORT = int(os.environ.get("APP_PORT", 8000))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# ── Database ──
DATABASE_URL = os.environ.get("DATABASE_PUBLIC_URL") or os.environ.get("DATABASE_URL", "sqlite:///./company_qa.db")

# ── Analytics ──
ANALYTICS_API_KEY = os.environ.get("ANALYTICS_API_KEY", "")
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.4"))