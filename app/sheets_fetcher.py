"""
Google Sheets data fetcher.

Authenticates via service account and fetches Q&A data from the client's
Google Sheet. Handles both minimal (question, answer) and enhanced
(+ link, category, keywords, active) column structures.
"""

import json
import logging
from typing import List, Dict, Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

from app.config import (
    GOOGLE_SERVICE_ACCOUNT_JSON,
    GOOGLE_SERVICE_ACCOUNT_FILE,
)

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
REQUIRED_COLUMNS = {"question", "answer"}


def get_sheets_service():
    """Initialize the Google Sheets API client using service account credentials."""
    if GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_SERVICE_ACCOUNT_JSON.strip():
        creds_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    elif GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_SERVICE_ACCOUNT_FILE.strip():
        creds = Credentials.from_service_account_file(
            GOOGLE_SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
    else:
        raise ValueError(
            "No Google credentials configured. "
            "Set GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_FILE."
        )

    return build("sheets", "v4", credentials=creds)


def fetch_sheet_data(
    spreadsheet_id: str,
    sheet_name: str = "Sheet1",
) -> List[Dict]:
    """
    Fetch all active rows from the Google Sheet.

    Returns a list of dicts keyed by header names. Handles:
    - Minimal columns: question, answer
    - Optional columns: link, category, keywords, active, id
    - Short rows (padded with empty strings)
    - Inactive rows (skipped if 'active' column exists and value != TRUE)
    """
    service = get_sheets_service()
    range_str = f"{sheet_name}!A:Z"

    try:
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=range_str)
            .execute()
        )
    except Exception as e:
        logger.error(f"Failed to fetch sheet data: {e}")
        raise

    rows = result.get("values", [])
    if len(rows) < 2:
        logger.warning("Sheet is empty or has only headers")
        return []

    # Parse headers (lowercase, stripped)
    headers = [h.strip().lower() for h in rows[0]]

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(headers)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise ValueError(f"Sheet is missing required columns: {missing}")

    has_active_column = "active" in headers
    records = []

    for i, row in enumerate(rows[1:], start=2):
        # Pad short rows
        padded = row + [""] * (len(headers) - len(row))
        record = dict(zip(headers, padded))
        record["_row_number"] = i

        # Skip inactive rows (only if 'active' column exists)
        if has_active_column:
            if record.get("active", "").strip().upper() != "TRUE":
                continue

        # Skip rows with empty question or answer
        question = record.get("question", "").strip()
        answer = record.get("answer", "").strip()
        if not question or not answer:
            continue

        records.append(record)

    logger.info(f"Fetched {len(records)} active records from sheet (rows scanned: {len(rows) - 1})")
    return records
