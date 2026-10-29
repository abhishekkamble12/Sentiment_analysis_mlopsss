# How to Deploy on Streamlit Community Cloud

This guide provides simple, step-by-step instructions to deploy the Tweet Sentiment Classifier on **Streamlit Community Cloud** for free.

---

## Prerequisites
- A GitHub account.
- This repository pushed to your GitHub (`https://github.com/abhishekkamble12/Sentiment_analysis_mlopsss`).
- A free account on [streamlit.io](https://share.streamlit.io).

---

## Step-by-Step Deployment

### 1. Sign in to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **Continue with GitHub** and authorize Streamlit.

### 2. Create a New App
1. Click the blue **Create app** (or **New app**) button in the top right.
2. Select **"I already have an app"**.

### 3. Configure Deployment Settings
Fill out the fields:
- **Repository:** `abhishekkamble12/Sentiment_analysis_mlopsss`
- **Branch:** `main`
- **Main file path:** `streamlit_app.py`
- **App URL (optional):** Choose your custom subdomain (e.g. `tweet-sentiment-ai.streamlit.app`).

### 4. Deploy!
Click **Deploy!**

Streamlit will automatically:
1. Spin up an isolated Linux container.
2. Read `requirements.txt` and install all required packages.
3. Launch `streamlit_app.py` at your public URL.

---

## Features Available in the Streamlit App

1. **Single Tweet Prediction**:
   - Classifies positive, neutral, or negative sentiment.
   - Shows calibrated confidence score and latency.
   - Visualizes probability distribution across all three classes.

2. **Bulk CSV Processing**:
   - Upload any CSV with a `text` column.
   - Downloads a ready-to-use template CSV.
   - Classifies thousands of rows with a progress bar and outputs a downloadable CSV with sentiment labels and confidence scores.

3. **Google Form Feedback Survey**:
   - Embeds your live Google Form directly inside Streamlit.
   - Users can fill the survey without leaving the app.

4. **Model Performance & Evaluation**:
   - Displays test metrics (Macro-F1, Accuracy, Kappa, ECE, Latency) across all models.
