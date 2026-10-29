import os
import re
import time
import json
import joblib
import pandas as pd
import asyncio
from typing import Dict, Any, List
from pathlib import Path
from huggingface_hub import InferenceClient
from tweet_sentiment_classifier.components.data_transformation import DataTransformation
from tweet_sentiment_classifier.config.configuration import ConfigurationManager
from tweet_sentiment_classifier import logger
from sklearn.base import BaseEstimator

from tweet_sentiment_classifier.config.env_config import get_settings

# Environment variable for HF Gemma token
settings = get_settings()
import os
HF_API_TOKEN = settings.HF_API_TOKEN or settings.HF_TOKEN or os.environ.get("HF_TOKEN", "")

INFERENCE_MODES = ("local", "llm", "hybrid")
HF_LLM_ID = "google/gemma-4-31b-it"
GROQ_LLM_ID = "llama-3.1-8b-instant"
SENTIMENT_MAP = {-1: "negative", 0: "neutral", 1: "positive"}

class PredictionPipeline:
    def __init__(self, inference_mode: str | None = None, confidence_threshold: float | None = None):
        self.inference_mode = (inference_mode or settings.INFERENCE_MODE).lower()
        if self.inference_mode not in INFERENCE_MODES:
            raise ValueError(f"INFERENCE_MODE must be one of {INFERENCE_MODES}, got {self.inference_mode!r}")
        self.confidence_threshold = (
            settings.HYBRID_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        )
        self.config_manager = ConfigurationManager()
        self.best_model_config = self.config_manager.get_best_model_config()
        self.metadata_path = Path(self.best_model_config.metadata_path)
        
        self.metadata = self._load_metadata()
        self.fast_model_name = self.metadata.get("tier_1_fast_fallback", "LogisticRegression")
        
        # Load preprocessing tool
        self.preprocessor_path = Path("artifacts/data_transformation/preprocessor.pkl")
        self.vectorizer = None
        if self.preprocessor_path.exists():
            self.vectorizer = joblib.load(self.preprocessor_path)
            
        # Load models
        self._load_models()
        self.hf_client = InferenceClient(HF_LLM_ID, token=HF_API_TOKEN) if HF_API_TOKEN else None
        
        # Dummy config for DataTransformation (we only need clean_text method)
        dummy_transform_config = self.config_manager.get_data_transformation_config()
        self.data_transformer = DataTransformation(dummy_transform_config)

    def _load_metadata(self) -> dict:
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_models(self):
        # Tier 1: Fast Model
        fast_model_path = Path("artifacts/model_trainer") / f"model_{self.fast_model_name}.pkl"
        if fast_model_path.exists():
            self.fast_model = joblib.load(fast_model_path)
        else:
            self.fast_model = None
            logger.warning(f"Fast model {self.fast_model_name} not found at {fast_model_path}")

        # Tier 2: DistilBERT
        distilbert_path = Path("artifacts/model_trainer/model_DistilBERT")
        import os as env_os
        env_os.environ["TF_USE_LEGACY_KERAS"] = "1"
        try:
            from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
            if distilbert_path.exists():
                model_source = str(distilbert_path)
            else:
                model_source = "distilbert-base-uncased-finetuned-sst-2-english"
                logger.info(f"Local DistilBERT not found, downloading {model_source} from HF...")
            
            self.db_tokenizer = AutoTokenizer.from_pretrained(model_source, use_safetensors=False)
            
            # If loading local or fallback to HF, native TF weights should be used.
            self.db_model = TFAutoModelForSequenceClassification.from_pretrained(model_source, use_safetensors=False)
        except Exception as e:
            self.db_tokenizer = None
            self.db_model = None
            logger.warning(f"Refinement model not loaded. Error: {e}")

    def _clean_text(self, text: str) -> str:
        return self.data_transformer.clean_text(text)

    def _predict_fast(self, cleaned_text: str) -> tuple[int, float]:
        if not self.fast_model or not self.vectorizer:
            raise RuntimeError("Fast model or vectorizer is not loaded.")
        
        vectorized = self.vectorizer.transform([cleaned_text])
        
        # Try predict_proba first (LogisticRegression)
        if hasattr(self.fast_model, "predict_proba"):
            probs = self.fast_model.predict_proba(vectorized)[0]
            pred_class = int(self.fast_model.classes_[probs.argmax()])
            confidence = float(probs.max())
        elif hasattr(self.fast_model, "decision_function"):
            # For LinearSVC, Ridge
            decision = self.fast_model.decision_function(vectorized)[0]
            # Convert decision distance to a pseudo-probability
            pred_class = int(self.fast_model.predict(vectorized)[0])
            import numpy as np
            confidence = float(np.exp(np.abs(decision)) / (1 + np.exp(np.abs(decision))))
            if isinstance(confidence, np.ndarray):
                confidence = float(confidence.max())
        else:
            pred_class = int(self.fast_model.predict(vectorized)[0])
            confidence = 1.0 # Unknown confidence
            
        return pred_class, confidence

    def _predict_distilbert(self, cleaned_text: str) -> tuple[int, float]:
        if not self.db_model or not self.db_tokenizer:
            raise RuntimeError("DistilBERT is not loaded.")
            
        import tensorflow as tf
        import numpy as np
        
        encodings = self.db_tokenizer([cleaned_text], truncation=True, padding=True, max_length=128, return_tensors="tf")
        logits = self.db_model(dict(encodings)).logits
        probs = tf.nn.softmax(logits, axis=-1)[0]
        
        pred_idx = tf.argmax(probs).numpy()
        confidence = float(probs[pred_idx])
        
        num_classes = logits.shape[-1]
        if num_classes == 2:
            pred_class = -1 if pred_idx == 0 else 1
        else:
            pred_class = int(pred_idx - 1.0) # Map back to -1, 0, 1
        
        return pred_class, confidence

    async def _predict_llm_async(self, text: str) -> dict | None:
        """Ask an external LLM for a label. Returns None on any failure; never invents a label."""
        prompt = (
            "Analyze the sentiment of the following tweet and return exactly -1 for negative, "
            "0 for neutral, or 1 for positive. Also provide a brief explanation. "
            f"Tweet: '{text}'.\nFormat: Label: [label]\nExplanation: [explanation]"
        )

        candidates = []  # (source, callable returning text)
        if self.hf_client:
            candidates.append((f"llm:hf/{HF_LLM_ID}", lambda: self.hf_client.text_generation(prompt, max_new_tokens=100)))
        if settings.GROQ_API_KEY:
            def _groq():
                from langchain_groq import ChatGroq
                from langchain_core.messages import HumanMessage
                llm = ChatGroq(model_name=GROQ_LLM_ID, api_key=settings.GROQ_API_KEY, temperature=0.1)
                return llm.invoke([HumanMessage(content=prompt)]).content
            candidates.append((f"llm:groq/{GROQ_LLM_ID}", _groq))

        for source, call in candidates:
            try:
                response_text = call()
            except Exception as e:
                logger.warning(f"{source} failed: {e}")
                continue
            parsed = self._parse_llm_response(response_text)
            if parsed is None:
                logger.warning(f"{source} returned an unparseable label: {str(response_text)[:120]!r}")
                continue
            label, explanation = parsed
            return {"label": label, "explanation": explanation, "source": source}
        return None

    @staticmethod
    def _parse_llm_response(response_text: str) -> tuple[int, str] | None:
        match = re.search(r"Label:\s*(-1|0|1)\b", response_text or "")
        if not match:
            return None
        explanation = response_text.split("Explanation:", 1)[1].strip() if "Explanation:" in response_text else ""
        return int(match.group(1)), explanation

    def _sync_predict_local_tiers(self, cleaned_text: str) -> dict:
        """Run each local model independently; a failing tier is reported, not masked."""
        out = {"fast": None, "distilbert": None, "errors": {}}
        try:
            out["fast"] = self._predict_fast(cleaned_text)
        except Exception as e:
            out["errors"]["fast"] = str(e)
            logger.warning(f"Fast model {self.fast_model_name} failed: {e}")
        try:
            out["distilbert"] = self._predict_distilbert(cleaned_text)
        except Exception as e:
            out["errors"]["distilbert"] = str(e)
            logger.warning(f"DistilBERT failed: {e}")
        return out

    async def predict(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        mode = self.inference_mode
        cleaned_text = self._clean_text(text)
        warnings: list[str] = []

        # 1. Local models (always computed: they are votes, and the fallback if the LLM fails)
        loop = asyncio.get_running_loop()
        try:
            local = await asyncio.wait_for(
                loop.run_in_executor(None, self._sync_predict_local_tiers, cleaned_text), timeout=5.0
            )
        except Exception as e:
            logger.error(f"Local tiers failed or timed out: {e}")
            local = {"fast": None, "distilbert": None, "errors": {"all": str(e)}}
        for tier in local["errors"]:
            warnings.append(f"local_{tier}_failed")

        fast_source = f"local:{self.fast_model_name.lower()}"
        votes: Dict[str, Dict[str, Any]] = {}
        local_decision = None  # (source, label, confidence)
        if local["fast"] is not None:
            label, conf = local["fast"]
            votes[fast_source] = {"label": SENTIMENT_MAP.get(label, "neutral"), "confidence": round(conf, 4)}
            local_decision = (fast_source, label, conf)
        if local["distilbert"] is not None:
            label, conf = local["distilbert"]
            votes["local:distilbert"] = {"label": SENTIMENT_MAP.get(label, "neutral"), "confidence": round(conf, 4)}
            if local_decision is None:
                local_decision = ("local:distilbert", label, conf)

        # 2. LLM, only when the mode asks for it
        llm_attempted = False
        llm_res = None
        if mode == "llm" or (
            mode == "hybrid" and (local_decision is None or local_decision[2] < self.confidence_threshold)
        ):
            llm_attempted = True
            llm_res = await self._predict_llm_async(text)
            if llm_res is None:
                warnings.append("llm_unavailable")
            else:
                votes[llm_res["source"]] = {"label": SENTIMENT_MAP[llm_res["label"]], "confidence": None}

        # 3. Decide, and report exactly who decided
        if llm_res is not None:
            source, label, confidence, explanation = llm_res["source"], llm_res["label"], None, llm_res["explanation"]
            fallback_triggered = mode == "hybrid"
        elif local_decision is not None:
            source, label, confidence = local_decision
            explanation = None
            fallback_triggered = llm_attempted  # LLM was wanted but failed, so we fell back to local
        else:
            raise RuntimeError("No model available: local models failed to load/run and no LLM answered.")

        latency = time.time() - start_time
        logger.info(f"mode={mode} decided_by={source} votes={votes}")

        return {
            "prediction": SENTIMENT_MAP.get(label, "neutral"),
            "confidence": None if confidence is None else round(confidence, 4),
            "latency_seconds": round(latency, 4),
            "latency_ms": round(latency * 1000, 2),
            "model_used": source,
            "decision_source": source,
            "inference_mode": mode,
            "fallback_triggered": fallback_triggered,
            "explanation": explanation,
            "votes": votes,
            "warnings": warnings,
        }
