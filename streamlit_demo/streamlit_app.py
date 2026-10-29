"""Streamlit Web Application for Tweet Sentiment Analysis MLOps Platform."""
import json
import time
from pathlib import Path

import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Resolve everything from the repo root so the working directory does not matter.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tweet_sentiment_classifier.components.data_transformation import clean_text  # noqa: E402
from tweet_sentiment_classifier.pipeline.explain import LinearExplainer  # noqa: E402

MAX_CSV_ROWS = 1000

# Configure page metadata
st.set_page_config(
    page_title="AI Tweet Sentiment Analytics",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .metric-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .sentiment-pos { color: #10b981; font-weight: bold; font-size: 1.5rem; }
    .sentiment-neg { color: #f43f5e; font-weight: bold; font-size: 1.5rem; }
    .sentiment-neu { color: #94a3b8; font-weight: bold; font-size: 1.5rem; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED MODEL LOADER -----------------
@st.cache_resource
def load_serving_pipeline():
    """Load the trained TF-IDF vectorizer and best calibrated model."""
    vec_path = ROOT / "artifacts/data_transformation/preprocessor.pkl"
    model_path = ROOT / "artifacts/best_model/best_model.pkl"
    meta_path = ROOT / "artifacts/best_model/best_model_metadata.json"

    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Fallback to model_trainer directory if best_model.pkl is missing
    if not model_path.exists():
        best_name = metadata.get("best_model_name", "LinearSVC")
        candidates = [
            ROOT / f"artifacts/model_trainer/model_{best_name}.pkl",
            ROOT / "artifacts/model_trainer/model.pkl",
            ROOT / "artifacts/model_trainer/model_LinearSVC.pkl",
            ROOT / "artifacts/model_trainer/model_LogisticRegression.pkl",
        ]
        for cand in candidates:
            if cand.exists():
                model_path = cand
                break

    if not vec_path.exists() or not model_path.exists():
        return None, None, metadata, model_path

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vectorizer, model, metadata, model_path


vectorizer, model, metadata, model_path = load_serving_pipeline()
LABEL_MAP = {-1: "Negative", 0: "Neutral", 1: "Positive"}
LABEL_ICONS = {"Negative": "🔴", "Neutral": "⚪", "Positive": "🟢"}

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🐦 Sentiment MLOps")
    st.markdown("**Production 3-Class Tweet Sentiment Classification**")
    
    st.divider()
    
    best_name = metadata.get("best_model_name", "LinearSVC")
    st.write("### ⚙️ Active Model")
    st.info(f"**Model:** {best_name}\n\n**Wrapper:** {type(model).__name__ if model is not None else "n/a"}\n\n**Classes:** Negative (-1), Neutral (0), Positive (1)")
    
    if metadata.get("metrics"):
        m = metadata["metrics"]
        st.write("### 📊 Benchmark Stats")
        c1, c2 = st.columns(2)
        c1.metric("Macro-F1", f"{m.get('macro_f1', 0):.4f}")
        c2.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
        st.caption(f"Median Latency: ~{m.get('latency_p50_ms', 0):.2f} ms")

    st.divider()
    st.markdown("Developed with ❤️ | [GitHub Repository](https://github.com/abhishekkamble12/Sentiment_analysis_mlopsss)")

# ----------------- MAIN INTERFACE -----------------
st.title("🚀 Tweet Sentiment Intelligence Platform")
st.markdown("Real-time NLP sentiment analysis trained with leak-free evaluation and probability calibration.")
st.warning(
    "**Read the scores with care.** The training labels come from the TextBlob lexicon, not from people "
    "(~91% of the tweets are about one topic). Scores measure how well a model imitates that lexicon, "
    "not how well it detects human-perceived sentiment. A human-labelled evaluation set is planned."
)

if not vectorizer or not model:
    st.error("⚠️ Model artifacts not found! Run the training pipeline (`python main.py`) first to generate the models.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time Prediction", 
    "📁 Batch CSV Processing", 
    "📝 Google Form Survey", 
    "📈 Model Benchmarks"
])

# ================= TAB 1: SINGLE PREDICTION =================
with tab1:
    st.subheader("Single Text Classification")
    
    col_input, col_preset = st.columns([3, 1])
    with col_preset:
        st.write("**Quick Test Samples:**")
        if st.button("🟢 Positive Sample"):
            st.session_state["tweet_text"] = "This is an amazing and fantastic achievement! Truly proud."
        if st.button("🔴 Negative Sample"):
            st.session_state["tweet_text"] = "Terrible experience, completely disgusted and deeply disappointed."
        if st.button("⚪ Neutral Sample"):
            st.session_state["tweet_text"] = "The schedule conference meeting will commence at 3:00 PM."
            
    with col_input:
        user_text = st.text_area(
            "Enter tweet text or public opinion:",
            value=st.session_state.get("tweet_text", ""),
            height=130,
            placeholder="Type or paste a tweet here..."
        )

    if st.button("🚀 Analyze Sentiment", type="primary", width="stretch"):
        if not user_text.strip():
            st.warning("Please enter text to analyze.")
        else:
            start_time = time.time()
            cleaned = clean_text(user_text)
            vec = vectorizer.transform([cleaned])
            probs = model.predict_proba(vec)[0]
            latency_ms = (time.time() - start_time) * 1000

            classes = model.classes_
            best_idx = probs.argmax()
            raw_class = classes[best_idx]
            label = LABEL_MAP.get(raw_class, "Neutral")
            conf = probs[best_idx]

            st.write("---")
            m1, m2, m3, m4 = st.columns(4)
            
            css_class = "sentiment-pos" if label == "Positive" else ("sentiment-neg" if label == "Negative" else "sentiment-neu")
            m1.markdown(f"<div class='metric-box'><div style='font-size:0.8rem; color:#94a3b8;'>PREDICTION</div><div class='{css_class}'>{LABEL_ICONS.get(label, '')} {label}</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-box'><div style='font-size:0.8rem; color:#94a3b8;'>CONFIDENCE</div><div style='font-size:1.5rem; font-weight:bold; color:white;'>{conf * 100:.1f}%</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-box'><div style='font-size:0.8rem; color:#94a3b8;'>LATENCY</div><div style='font-size:1.5rem; font-weight:bold; color:#818cf8;'>{latency_ms:.1f} ms</div></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-box'><div style='font-size:0.8rem; color:#94a3b8;'>DECIDER</div><div style='font-size:1.2rem; font-weight:bold; color:#cbd5e1;'>local:{best_name.lower()}</div></div>", unsafe_allow_html=True)

            st.write("#### Class Probabilities Breakdown")
            prob_dict = {LABEL_MAP.get(c, str(c)): float(p) for c, p in zip(classes, probs)}
            chart_df = pd.DataFrame(list(prob_dict.items()), columns=["Sentiment", "Probability"]).set_index("Sentiment")
            st.bar_chart(chart_df)

            explainer = LinearExplainer.maybe(model, vectorizer)
            if explainer is not None:
                out = explainer.explain(user_text, cleaned, raw_class)
                st.write("#### Words behind this prediction")
                rows = [{"term": t["term"], "weight": t["weight"], "effect": f"toward {label.lower()}"}
                        for t in out["supporting"]]
                rows += [{"term": t["term"], "weight": t["weight"], "effect": f"against {label.lower()}"}
                         for t in out["opposing"]]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.caption(out["note"])

# ================= TAB 2: BATCH CSV PROCESSING =================
with tab2:
    st.subheader("Bulk CSV Sentiment Analysis")
    st.markdown("Upload a CSV file containing a `text` column to classify all entries in bulk.")

    sample_csv = "text\nI absolutely loved this product, works flawlessly!\nWorst customer service experience I have ever had.\nThe package arrived at 2pm on Monday.\n"
    st.download_button(
        "📄 Download Sample Template CSV",
        data=sample_csv,
        file_name="sample_tweets.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("The CSV must contain a column named 'text'.")
            elif len(df) > MAX_CSV_ROWS:
                st.error(f"Too many rows ({len(df)}). The demo accepts at most {MAX_CSV_ROWS}.")
            else:
                st.write(f"Loaded **{len(df)} rows**. Preview:")
                st.dataframe(df.head(5), width="stretch")

                if st.button("⚡ Run Bulk Analysis", type="primary"):
                    with st.spinner(f"Analyzing {len(df)} rows..."):
                        t0 = time.time()
                        texts = df["text"].fillna("").astype(str).map(clean_text)
                        probs = model.predict_proba(vectorizer.transform(texts))
                        classes = model.classes_
                        pred_indices = probs.argmax(axis=1)
                        out = df.copy()
                        out["sentiment"] = [LABEL_MAP.get(classes[i], "Neutral") for i in pred_indices]
                        out["confidence"] = np.round(probs[np.arange(len(out)), pred_indices], 4)
                        # Kept in session state so the results survive the rerun a download click causes.
                        st.session_state["batch_result"] = (out, time.time() - t0)
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

    if "batch_result" in st.session_state:
        out, elapsed = st.session_state["batch_result"]
        st.success(f"Processed {len(out)} rows in {elapsed:.2f} seconds ({len(out)/max(elapsed, 0.001):.0f} rows/sec).")
        st.dataframe(out.head(25), width="stretch")

        c1, c2 = st.columns([1, 2])
        counts = out["sentiment"].value_counts()
        with c1:
            st.write("#### Sentiment Breakdown")
            st.dataframe(counts)
        with c2:
            st.bar_chart(counts)

        st.download_button(
            label="📥 Download Analyzed CSV Results",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_predictions.csv",
            mime="text/csv",
            type="primary",
        )

# ================= TAB 3: GOOGLE FORM SURVEY =================
with tab3:
    st.subheader("Google Form Survey & Live Feedback")
    st.markdown("Embed your Google Form to collect live feedback directly inside the Streamlit app:")

    form_url = st.text_input(
        "Google Form Embed URL:",
        placeholder="https://docs.google.com/forms/d/e/YOUR_FORM_ID/viewform?embedded=true"
    )

    if form_url.strip():
        embed_src = form_url.strip()
        if "/viewform" in embed_src and "embedded=true" not in embed_src:
            embed_src += ("&embedded=true" if "?" in embed_src else "?embedded=true")
        st.components.v1.iframe(embed_src, height=800, scrolling=True)
    else:
        st.info("💡 Paste your Google Form embed URL above to display the live form here. You can find this under **Google Form &rarr; Send &rarr; Embed `<>`**.")

# ================= TAB 4: MODEL BENCHMARKS =================
with tab4:
    st.subheader("Model Evaluation & Comparison Benchmarks")
    st.markdown("All metrics are computed on a held-out test split of **31,917 tweets** with 0% text leakage.")

    eval_path = ROOT / "artifacts/evaluation/results.csv"
    if eval_path.exists():
        results_df = pd.read_csv(eval_path)
        st.dataframe(results_df, width="stretch")
        st.caption(
            "TextBlob generated the labels, so its ~1.0 score only confirms the labels. It is not a competitor "
            "and is left out of the chart. All scores are *silver* (agreement with TextBlob), not human accuracy."
        )

        st.write("#### Macro-F1 Comparison")
        chart_data = results_df[results_df["model"] != "TextBlob"][["model", "macro_f1"]].set_index("model")
        st.bar_chart(chart_data)
    else:
        st.info("Evaluation results not found in `artifacts/evaluation/results.csv`.")

