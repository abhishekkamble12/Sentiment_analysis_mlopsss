"""Model and evaluation information, read from the artifact files."""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Depends

from tweet_sentiment_classifier.app.deps import get_container

logger = logging.getLogger("api")
router = APIRouter(tags=["models"])

def _load_best_model_meta() -> Dict[str, Any]:
    try:
        return json.loads(Path("artifacts/best_model/best_model_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


BASELINE_MODELS = {"Majority", "TextBlob", "VADER"}


def _optional_float(value):
    return None if value is None or pd.isna(value) else round(float(value), 4)


@router.get("/models/experiments")
def get_experiments():
    """Results of the latest single evaluation run (artifacts/evaluation/results.csv)."""
    summary_path = Path("artifacts/evaluation/results.csv")
    if not summary_path.exists():
        return {"experiments": [], "run_id": None}

    try:
        df = pd.read_csv(summary_path)
        run_id = None
        results_json = summary_path.with_suffix(".json")
        if results_json.exists():
            run_id = json.loads(results_json.read_text(encoding="utf-8")).get("run_id")

        # Mirror what PredictionPipeline actually serves (see prediction_pipeline.py)
        meta = _load_best_model_meta()
        best = meta.get("best_model_name")
        fast = meta.get("tier_1_fast_fallback", "LogisticRegression")

        experiments = []
        for _, row in df.iterrows():
            name = str(row["model"])
            labels = []
            if name in BASELINE_MODELS:
                labels.append("Baseline (label source)" if name == "TextBlob" else "Baseline")
            if name == fast:
                labels.append("Serving (fast tier)")
            if name == best:
                labels.append("Best by macro-F1")
            experiments.append({
                "run_id": run_id,
                "model_name": name,
                "n_test": int(row["n"]),
                "accuracy": round(float(row["accuracy"]), 4),
                "f1_macro": round(float(row["macro_f1"]), 4),
                "f1_weighted": round(float(row["weighted_f1"]), 4),
                "ci_low": _optional_float(row.get("macro_f1_ci_low")),
                "ci_high": _optional_float(row.get("macro_f1_ci_high")),
                "ece": _optional_float(row.get("ece")),
                "latency_p50_ms": _optional_float(row.get("latency_p50_ms")),
                "status": " + ".join(labels) or "Evaluated",
            })
        return {"experiments": experiments, "run_id": run_id}
    except Exception as e:
        logger.error(f"Error reading experiments: {e}")
        return {"experiments": [], "error": str(e)}



@router.get("/models/info")
def get_model_info(c=Depends(get_container)):
    """What the API is actually serving: model name, inference mode, transformer status, label map."""
    meta = _load_best_model_meta()
    pipeline = c.pipeline
    ready = pipeline is not None
    return {
        "pipeline_ready": ready,
        "best_model_name": meta.get("best_model_name"),
        "fast_model_name": meta.get("tier_1_fast_fallback", "LogisticRegression"),
        "inference_mode": getattr(pipeline, "inference_mode", None) if ready else None,
        "transformer_enabled": bool(getattr(pipeline, "db_model", None)) if ready else False,
        "label_mapping": {"-1": "negative", "0": "neutral", "1": "positive"},
        "best_metrics": meta.get("metrics"),
        "run_id": meta.get("run_id"),
    }


RESULTS_DIR = Path("artifacts/evaluation")
LABEL_WORDS = {-1: "negative", 0: "neutral", 1: "positive"}


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _model_table(run: Dict[str, Any]) -> list:
    rows = []
    for name, r in run.get("results", {}).items():
        ci = r.get("macro_f1_ci") or (None, None)
        rows.append({"model": name, "macro_f1": r["macro_f1"], "ci_low": ci[0], "ci_high": ci[1],
                     "accuracy": r["accuracy"], "kappa": r["kappa"], "ece": r.get("ece"),
                     "baseline": name in BASELINE_MODELS})
    return rows


def _run_summary(run):
    if not run:
        return None
    split = run.get("split", {})
    return {"run_id": run.get("run_id"), "n_test": split.get("n_test"), "label_source": split.get("label_source"),
            "overlap_pct": split.get("test_texts_also_in_train_pct"), "weights_column": split.get("weights_column"),
            "models": _model_table(run), "comparisons": run.get("comparisons", [])}


def _lab_run(run):
    """Full per-model detail for the Model Lab page: per-class F1, confusion matrix and latency.

    Read straight from the evaluation artifacts; nothing is invented or rounded away.
    """
    if not run:
        return None
    split = run.get("split", {})
    models = []
    for name, r in run.get("results", {}).items():
        ci = r.get("macro_f1_ci") or (None, None)
        models.append({
            "model": name,
            "accuracy": r.get("accuracy"),
            "macro_f1": r.get("macro_f1"),
            "weighted_f1": r.get("weighted_f1"),
            "kappa": r.get("kappa"),
            "ece": r.get("ece"),
            "ci_low": ci[0], "ci_high": ci[1],
            "baseline": name in BASELINE_MODELS,
            "per_class": r.get("per_class"),
            "confusion_matrix": r.get("confusion_matrix"),
            "latency": r.get("latency"),
        })
    return {"run_id": run.get("run_id"), "split": split, "models": models,
            "comparisons": run.get("comparisons", [])}



@router.get("/models/results")
def get_results():
    """Everything on the results page, read from artifact files (nothing is hardcoded)."""
    seeds = _read_json(RESULTS_DIR / "seed_study.json")
    tuning = _read_json(RESULTS_DIR / "tuning.json")
    budget = _read_json(Path("artifacts/experiments/gold_budget/results.json"))

    disagreements = []
    gold_predictions = RESULTS_DIR / "gold" / "predictions.csv"
    if gold_predictions.exists():
        pred = pd.read_csv(gold_predictions)
        if "pred_TextBlob" in pred:
            diff = pred[pred["pred_TextBlob"] != pred["label"]].head(25)
            disagreements = [{"text": r["text"], "human": LABEL_WORDS.get(int(r["label"])),
                              "textblob": LABEL_WORDS.get(int(r["pred_TextBlob"]))} for _, r in diff.iterrows()]

    return {
        "silver": _run_summary(_read_json(RESULTS_DIR / "results.json")),
        "gold": _run_summary(_read_json(RESULTS_DIR / "gold" / "results.json")),
        "seed_study": None if not seeds else {
            "seeds": seeds["seeds"],
            "models": [{"model": m, "mean": s["macro_f1"]["mean"], "sd": s["macro_f1"]["sd"],
                        "min": s["macro_f1"]["min"], "max": s["macro_f1"]["max"]} for m, s in seeds["summary"].items()]},
        "tuning_best": None if not tuning else tuning.get("best"),
        "gold_budget": None if not budget else budget.get("summary"),
        "disagreements": disagreements,
    }


@router.get("/models/lab")
def get_model_lab():
    """Real evaluation detail for the Model Lab page: per-class F1, confusion matrices, latency,
    pairwise comparisons and the silver/gold split, read straight from the evaluation artifacts."""
    return {
        "silver": _lab_run(_read_json(RESULTS_DIR / "results.json")),
        "gold": _lab_run(_read_json(RESULTS_DIR / "gold" / "results.json")),
        "label_order": ["negative", "neutral", "positive"],
        "best_model_name": _load_best_model_meta().get("best_model_name"),
        "limitation": ("Silver metrics use TextBlob-generated labels, not human annotations: they measure how "
                       "well a model imitates that lexicon, not human-perceived sentiment. Gold (human-labelled) "
                       "results appear only once that set exists."),
    }


