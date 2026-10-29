# Google Form Integration Guide

This guide explains how to connect any Google Form to the Tweet Sentiment Classification system so that user responses are automatically analyzed for sentiment in real-time and stored in the monitoring database.

---

## Architecture Overview

```
[ User Submits Google Form ]
             │
             ▼
[ Google Sheets Response Row ]
             │
             ▼ (Google Apps Script: On Form Submit Trigger)
[ HTTP POST to /api/form-submission ]
             │
             ▼
[ Sentiment Classifier (LinearSVC / LLM) ]
             │
             ├──► Saves result to database (`prediction_logs`)
             ├──► Shows on Real-time Monitoring & Analytics Dashboards
             └──► Writes Sentiment & Confidence back to Google Sheets
```

---

## Step 1: Create Your Google Form

1. Go to [Google Forms](https://forms.google.com) and create a new form (e.g., *"Customer Sentiment & Feedback Survey"*).
2. Add your questions:
   - **Question 1:** Name or Email (*Short answer, optional*)
   - **Question 2:** Feedback or Tweet Text (*Paragraph or Short answer, required*)
3. To embed the form inside your web dashboard:
   - Click **Send** (top right) &rarr; Click the embed icon **`<>`**.
   - Copy the iframe `src` link (e.g., `https://docs.google.com/forms/d/e/.../viewform?embedded=true`).
   - Open your web dashboard at `http://127.0.0.1:8000/feedback`.
   - Paste the link into the **Google Form URL** bar and click **Update Embed**.

---

## Step 2: Connect Form to Google Sheets

1. In Google Forms, click the **Responses** tab.
2. Click **Link to Sheets** (green spreadsheet icon).
3. Select **Create a new spreadsheet** &rarr; **Create**.
4. Note your column layout. By default:
   - **Column A:** Timestamp
   - **Column B:** Name / Respondent
   - **Column C:** Feedback Text
   - **Column D:** *(Will store Sentiment)*
   - **Column E:** *(Will store Confidence)*

---

## Step 3: Add Google Apps Script Trigger

1. In the linked Google Sheet, click **Extensions** &rarr; **Apps Script**.
2. Delete any existing code in the editor and paste the following script:

```javascript
function onFormSubmit(e) {
  // e.values contains the submitted row as an array of strings:
  // [Timestamp, Respondent, FeedbackText, ...]
  // Adjust the column indices below to match your questions (0-indexed):
  var timestamp = e.values[0];
  var respondent = e.values[1] || "Anonymous";
  var feedbackText = e.values[2]; // Change index if feedback is in a different column

  if (!feedbackText || feedbackText.trim() === "") {
    return;
  }

  // Set this to your server's public URL (or ngrok URL during local testing)
  var API_URL = "https://YOUR_DOMAIN_OR_NGROK_URL/api/form-submission";

  var payload = {
    feedback_text: feedbackText,
    respondent: respondent
  };

  var options = {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  try {
    var response = UrlFetchApp.fetch(API_URL, options);
    var statusCode = response.getResponseCode();

    if (statusCode === 200) {
      var data = JSON.parse(response.getContentText());

      // Write predicted sentiment and confidence back into the next columns
      var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
      var lastRow = sheet.getLastRow();
      
      // Column D (4): Sentiment ("positive", "negative", "neutral")
      sheet.getRange(lastRow, 4).setValue(data.sentiment);
      
      // Column E (5): Confidence (e.g. 0.9412)
      if (data.confidence !== null && data.confidence !== undefined) {
        sheet.getRange(lastRow, 5).setValue(Math.round(data.confidence * 10000) / 100 + "%");
      } else {
        sheet.getRange(lastRow, 5).setValue("N/A (LLM)");
      }
    } else {
      Logger.log("API returned status: " + statusCode + " - " + response.getContentText());
    }
  } catch (error) {
    Logger.log("Failed to send webhook to Sentiment API: " + error);
  }
}
```

3. Save the project (click the **disk icon** or `Ctrl + S`).

---

## Step 4: Configure the "On form submit" Trigger

1. In the Apps Script sidebar on the left, click the **Triggers** icon (clock / alarm clock).
2. Click **+ Add Trigger** (bottom right).
3. Set the following fields:
   - **Choose which function to run:** `onFormSubmit`
   - **Choose which deployment should run:** `Head`
   - **Select event source:** `From spreadsheet`
   - **Select event type:** `On form submit`
4. Click **Save** and grant permissions if prompted by Google.

---

## Step 5: Testing Locally with ngrok

During local development, your FastAPI server runs on `http://127.0.0.1:8000`. Google's cloud servers cannot reach `localhost` directly.

To route Google Form submissions to your local machine:
1. Run [ngrok](https://ngrok.com) in a terminal:
   ```bash
   ngrok http 8000
   ```
2. Copy the forwarding HTTPS URL (e.g. `https://1234-56-78.ngrok-free.app`).
3. Replace `https://YOUR_DOMAIN_OR_NGROK_URL` in the Apps Script with your forwarding URL:
   ```javascript
   var API_URL = "https://1234-56-78.ngrok-free.app/api/form-submission";
   ```
4. Submit your Google Form!
   - The feedback will be analyzed and logged into `monitoring.db`.
   - Your Google Sheet will immediately show the predicted sentiment and confidence.
   - The dashboard at `http://127.0.0.1:8000/monitoring-dashboard` and `http://127.0.0.1:8000/analytics-dashboard` will automatically reflect the new submission.
