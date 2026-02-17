/**
 * Google Apps Script — Webhook Trigger for Company Q&A Bot
 *
 * This script is installed in the client's Google Sheet.
 * When anyone edits the sheet, it sends a POST request to the
 * backend webhook endpoint, which triggers a knowledge base re-index.
 *
 * SETUP:
 * 1. In the Google Sheet, go to Extensions → Apps Script
 * 2. Replace the default code with this entire file
 * 3. Update WEBHOOK_URL with your actual backend URL
 * 4. Update WEBHOOK_SECRET to match your backend's SHEETS_WEBHOOK_SECRET env var
 * 5. Click the clock icon (Triggers) → Add Trigger:
 *    - Function: onSheetEdit
 *    - Event source: From spreadsheet
 *    - Event type: On edit
 * 6. Save and authorize when prompted
 */

// ══════════════════════════════════════════════════
// CONFIGURATION — Update these values
// ══════════════════════════════════════════════════

const WEBHOOK_URL = "https://your-qa-bot-domain.com/api/webhooks/sheets-update";
const WEBHOOK_SECRET = "your-shared-secret-here";

// Debounce window in seconds (prevents duplicate triggers on bulk edits)
const DEBOUNCE_SECONDS = 30;


// ══════════════════════════════════════════════════
// TRIGGER FUNCTION — Do not rename
// ══════════════════════════════════════════════════

function onSheetEdit(e) {
  // Debounce: skip if last trigger was within the debounce window
  const cache = CacheService.getScriptCache();
  const lastRun = cache.get("lastWebhookTrigger");
  if (lastRun) {
    console.log("Debounced — skipping webhook (last trigger was < " + DEBOUNCE_SECONDS + "s ago)");
    return;
  }

  // Set debounce lock
  cache.put("lastWebhookTrigger", "true", DEBOUNCE_SECONDS);

  // Build payload
  const sheet = SpreadsheetApp.getActiveSpreadsheet();
  const payload = {
    spreadsheet_id: sheet.getId(),
    sheet_name: e.source.getActiveSheet().getName(),
    timestamp: new Date().toISOString(),
    secret: WEBHOOK_SECRET
  };

  const options = {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  try {
    const response = UrlFetchApp.fetch(WEBHOOK_URL, options);
    const code = response.getResponseCode();

    if (code === 200) {
      console.log("Webhook sent successfully. Response: " + response.getContentText());
    } else {
      console.error("Webhook failed with status " + code + ": " + response.getContentText());
    }
  } catch (err) {
    console.error("Webhook request failed: " + err.toString());
  }
}


// ══════════════════════════════════════════════════
// MANUAL TEST FUNCTION
// ══════════════════════════════════════════════════

/**
 * Run this manually from the Apps Script editor to test
 * the webhook without editing the sheet.
 * Go to: Run → testWebhook
 */
function testWebhook() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet();

  const payload = {
    spreadsheet_id: sheet.getId(),
    sheet_name: sheet.getActiveSheet().getName(),
    timestamp: new Date().toISOString(),
    secret: WEBHOOK_SECRET
  };

  const options = {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  try {
    const response = UrlFetchApp.fetch(WEBHOOK_URL, options);
    console.log("Status: " + response.getResponseCode());
    console.log("Response: " + response.getContentText());
  } catch (err) {
    console.error("Test webhook failed: " + err.toString());
  }
}
