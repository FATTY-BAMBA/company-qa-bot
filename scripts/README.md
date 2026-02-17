# Company Q&A Bot

AI-powered customer service chatbot with a Google Sheets-managed knowledge base. Built for the Class Genius platform (Contract 3, Deliverable 3).

## Architecture

```
┌─────────────────┐     webhook      ┌──────────────────────────────┐
│  Google Sheet    │ ──────────────►  │  FastAPI Service             │
│  (client edits)  │                  │                              │
└─────────────────┘                  │  /api/webhooks/sheets-update │
                                     │  /api/chat                   │
                                     │  /api/admin/reindex          │
┌─────────────────┐   query/embed    │  /api/health                 │
│  Company Website │ ──────────────►  │                              │
│  (visitor chat)  │ ◄────────────── │                              │
└─────────────────┘   response       └──────┬───────────┬───────────┘
                                            │           │
                                     ┌──────▼──┐  ┌─────▼─────┐
                                     │ Pinecone │  │  OpenAI   │
                                     │ (vectors)│  │ (embed +  │
                                     │          │  │  chat)    │
                                     └─────────┘  └───────────┘
```

## Project Structure

```
company-qa-bot/
├── app/
│   ├── __init__.py
│   ├── config.py              # Environment variables
│   ├── main.py                # FastAPI app + all endpoints
│   ├── sheets_fetcher.py      # Google Sheets API integration
│   ├── indexer.py             # Chunking, embedding, Pinecone upsert
│   ├── chat.py                # RAG retrieval + LLM response
│   └── scheduled_sync.py     # Hourly fallback sync with hashing
├── scripts/
│   └── google_apps_script.js  # Apps Script for Google Sheet webhook
├── .env.example               # Environment variable template
├── requirements.txt           # Python dependencies
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual values
```

### 3. Run the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Initial index

Trigger the first indexing of your Google Sheet data:

```bash
curl -X POST http://localhost:8000/api/admin/reindex
```

### 5. Set up Google Apps Script

See `scripts/google_apps_script.js` for the webhook trigger code.
Install it in the Google Sheet via Extensions → Apps Script.

### 6. Set up scheduled sync (optional fallback)

```bash
# Crontab: run every hour
0 * * * * cd /path/to/company-qa-bot && python -m app.scheduled_sync
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Visitor chat query |
| POST | `/api/webhooks/sheets-update` | Google Sheets edit webhook |
| POST | `/api/admin/reindex` | Manual re-index trigger |
| GET | `/api/health` | Health check |

## Chat Request Example

```json
{
  "query": "你們有 Python 課程嗎？",
  "session_id": "optional-session-id"
}
```

## Chat Response Example

```json
{
  "answer": "有的！我們提供多門 Python 課程...",
  "sources": [
    {
      "row_number": 3,
      "question": "你們有 Python 相關的課程嗎？",
      "relevance_score": 0.92,
      "link": "https://www.example.com/courses/python"
    }
  ],
  "confidence": 0.89,
  "session_id": "abc-123",
  "timestamp": "2026-02-16T09:30:00Z",
  "latency_seconds": 1.2,
  "matches_found": 3
}
```
