import os
import time
import json
import joblib
import numpy as np
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
from tweet_sentiment_classifier.pipeline.explain import LinearExplainer
from tweet_sentiment_classifier.utils.llm_safety import build_sentiment_prompt, parse_sentiment_json

# Environment variable for HF Gemma token
settings = get_settings()
import os
HF_API_TOKEN = settings.HF_API_TOKEN or settings.HF_TOKEN or os.environ.get("HF_TOKEN", "")

INFERENCE_MODES = ("local", "llm", "hybrid")
HF_LLM_ID = "google/gemma-4-31b-it"
GROQ_LLM_ID = "llama-3.1-8b-instant"
SENTIMENT_MAP = {-1: "negative", 0: "neutral", 1: "positive"}

def transformer_num_labels(model_dir: Path) -> int | None:
    """Number of output classes declared by a HF checkpoint's config.json (None if unreadable)."""
    try:
        with open(Path(model_dir) / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    if "id2label" in cfg:
        return len(cfg["id2label"])
    return cfg.get("num_labels")


class PredictionPipeline:
    def __init__(self, inference_mode: str | None = None, confidence_threshold: float | None = None):
        self.inference_mode = (inference_mode or settings.INFERENCE_MODE).lower()
        if self.inference_mode not in INFERENCE_MODES:
            raise ValueError(f"INFERENCE_MODE must be one of {INFERENCE_MODES}, got {self.inference_mode!r}")
        self.confidence_threshold = (
            settings.HYBRID_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        )
        self.abstain_threshold = settings.ABSTAIN_THRESHOLD
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
        self.explainer = LinearExplainer.maybe(self.fast_model, self.vectorizer) if self.vectorizer else None
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

        # Tier 2: fine-tuned 3-class transformer (PyTorch). Never substitute a pretrained
        # model with a different label space: a 2-class sentiment model cannot output neutral.
        self.db_tokenizer = None
        self.db_model = None
        tier2_path = Path("artifacts/model_trainer/model_DistilBERT")
        if not tier2_path.exists():
            logger.warning(f"Transformer tier disabled: no fine-tuned model at {tier2_path}")
            return
        num_labels = transformer_num_labels(tier2_path)
        if num_labels != 3:
            logger.warning(f"Transformer tier disabled: {tier2_path} has {num_labels} labels, expected 3")
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.db_tokenizer = AutoTokenizer.from_pretrained(str(tier2_path))
            self.db_model = AutoModelForSequenceClassification.from_pretrained(str(tier2_path)).eval()
        except Exception as e:
            self.db_tokenizer = None
            self.db_model = None
            logger.warning(f"Transformer tier disabled: failed to load {tier2_path}: {e}")

    def _clean_text(self, text: str) -> str:
        return self.data_transformer.clean_text(text)

    def _predict_fast(self, cleaned_text: str) -> tuple[int, float]:
        if not self.fast_model or not self.vectorizer:
            raise RuntimeError("Fast model or vectorizer is not loaded.")
        
        vectorized = self.vectorizer.transform([cleaned_text])
        
        if not hasattr(self.fast_model, "predict_proba"):
            # Margin models (LinearSVC, Ridge) are calibrated at training time. An old uncalibrated
            # pickle must be retrained: a made-up confidence would be reported as if it were real.
            raise RuntimeError(
                f"{type(self.fast_model).__name__} has no predict_proba; "
                "retrain it (margin models are wrapped in CalibratedClassifierCV)."
            )

        probs = self.fast_model.predict_proba(vectorized)[0]
        classes = np.asarray(self.fast_model.classes_)
        if set(classes.tolist()) == {0, 1, 2}:  # XGBoost is trained on labels shifted by +1
            classes = classes - 1
        return int(classes[probs.argmax()]), float(probs.max())

    def _predict_distilbert(self, cleaned_text: str) -> tuple[int, float]:
        if not self.db_model or not self.db_tokenizer:
            raise RuntimeError("Transformer tier is not loaded.")

        import torch

        encodings = self.db_tokenizer([cleaned_text], truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = self.db_model(**encodings).logits
        probs = torch.softmax(logits, dim=-1)[0]
        if probs.shape[-1] != 3:
            raise RuntimeError(f"Transformer tier produced {probs.shape[-1]} classes, expected 3.")

        pred_idx = int(probs.argmax())
        return pred_idx - 1, float(probs[pred_idx])  # training maps sentiment -1,0,1 -> 0,1,2

    async def _predict_llm_async(self, text: str) -> dict | None:
        """Ask an external LLM for a label. Returns None on any failure; never invents a label.

        The SDK calls are blocking, so they run in a worker thread to keep the event loop free.
        """
        prompt = build_sentiment_prompt(text)

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
                response_text = await asyncio.to_thread(call)
            except Exception as e:
                logger.warning(f"{source} failed: {e}")
                continue
            parsed = parse_sentiment_json(response_text)
            if parsed is None:
                logger.warning(f"{source} returned an invalid classification: {str(response_text)[:120]!r}")
                continue
            label, explanation = parsed
            return {"label": label, "explanation": explanation, "source": source}
        return None

    def _sync_predict_local_tiers(self, cleaned_text: str) -> dict:
        """Run each local model independently; a failing tier is reported, not masked."""
        out = {"fast": None, "distilbert": None, "errors": {}}
        try:
            out["fast"] = self._predict_fast(cleaned_text)
        except Exception as e:
            out["errors"]["fast"] = str(e)
            logger.warning(f"Fast model {self.fast_model_name} failed: {e}")
        if self.db_model is None:  # tier disabled (no 3-label checkpoint): not a failure, so no warning
            return out
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

        contributions = None
        explainer = getattr(self, "explainer", None)
        if source == fast_source and explainer is not None:
            try:
                contributions = explainer.explain(text, cleaned_text, label)
            except Exception as e:
                logger.warning(f"Explanation failed: {e}")

        prediction = SENTIMENT_MAP.get(label, "neutral")
        abstain = getattr(self, "abstain_threshold", None)
        if abstain is not None and confidence is not None and confidence < abstain:
            prediction = "uncertain"  # a local decision below the validated threshold is not reported as a label
            warnings.append("abstained_low_confidence")

        latency = time.time() - start_time
        logger.info(f"mode={mode} decided_by={source} votes={votes}")

        return {
            "prediction": prediction,
            "confidence": None if confidence is None else round(confidence, 4),
            "latency_seconds": round(latency, 4),
            "latency_ms": round(latency * 1000, 2),
            "model_used": source,
            "decision_source": source,
            "inference_mode": mode,
            "fallback_triggered": fallback_triggered,
            "explanation": explanation,
            "contributions": contributions,
            "votes": votes,
            "warnings": warnings,
        }
