"""
Scheduled sync — fallback re-indexing.

Runs on a schedule (e.g., hourly via cron) to catch any webhook misses.
Uses content hashing to skip re-indexing when nothing has changed.
"""

import hashlib
import logging
import os

from app.config import COMPANY_QA_SPREADSHEET_ID, COMPANY_QA_SHEET_NAME
from app.sheets_fetcher import fetch_sheet_data
from app.indexer import reindex_company_qa

logger = logging.getLogger(__name__)

HASH_FILE = "/tmp/company_qa_last_hash.txt"


def compute_sheet_hash(spreadsheet_id: str, sheet_name: str) -> str:
    """Hash the sheet content to detect changes."""
    records = fetch_sheet_data(spreadsheet_id, sheet_name)
    content = str(sorted([str(r) for r in records]))
    return hashlib.sha256(content.encode()).hexdigest()


def get_last_hash() -> str | None:
    """Read the last known content hash from disk."""
    try:
        with open(HASH_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def save_hash(content_hash: str):
    """Save the current content hash to disk."""
    with open(HASH_FILE, "w") as f:
        f.write(content_hash)


def sync_if_changed() -> dict:
    """
    Check if the sheet content has changed since the last sync.
    If yes, trigger a full re-index. If no, skip.

    Returns a summary dict.
    """
    logger.info("Scheduled sync: checking for sheet changes...")

    current_hash = compute_sheet_hash(
        COMPANY_QA_SPREADSHEET_ID, COMPANY_QA_SHEET_NAME
    )
    last_hash = get_last_hash()

    if current_hash == last_hash:
        logger.info("No changes detected. Skipping re-index.")
        return {"status": "skipped", "reason": "no_changes"}

    logger.info("Changes detected. Triggering re-index...")
    result = reindex_company_qa(
        COMPANY_QA_SPREADSHEET_ID, COMPANY_QA_SHEET_NAME
    )

    save_hash(current_hash)
    result["triggered_by"] = "scheduled_sync"
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = sync_if_changed()
    print(result)
