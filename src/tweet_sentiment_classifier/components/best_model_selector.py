import json
import shutil
import pandas as pd
from pathlib import Path
from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.config.config_entity import BestModelConfig
from tweet_sentiment_classifier.utils.common import save_json

# Training-free baselines are reference points, never candidates. TextBlob in particular generated the
# silver labels, so on silver data it would always "win".
BASELINES = {"Majority", "TextBlob", "VADER"}
RESULTS_CSV = Path("artifacts/evaluation/results.csv")
MODEL_DIR = Path("artifacts/model_trainer")


class BestModelSelector:
    def __init__(self, config: BestModelConfig, results_csv: Path = RESULTS_CSV, model_dir: Path = MODEL_DIR):
        self.config = config
        self.results_csv = Path(results_csv)
        self.model_dir = Path(model_dir)

    def select_best_model(self) -> str:
        """Pick the trained model with the best test macro-F1 (ties: lower latency) from one evaluation run."""
        if not self.results_csv.exists():
            raise FileNotFoundError(f"{self.results_csv} not found; run tweet_sentiment_classifier.evaluation.run first")

        df = pd.read_csv(self.results_csv)
        candidates = df[~df["model"].isin(BASELINES)].copy()
        if candidates.empty:
            raise ValueError(f"{self.results_csv} has no trained models, only baselines")
        candidates["latency_p50_ms"] = candidates["latency_p50_ms"].fillna(float("inf"))
        best_row = candidates.sort_values(["macro_f1", "latency_p50_ms"], ascending=[False, True]).iloc[0]
        best = str(best_row["model"])
        logger.info(f"Best trained model by macro-F1: {best} ({best_row['macro_f1']:.4f})")

        src = self.model_dir / f"model_{best}.pkl"
        dest = Path(self.config.model_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dest)
        else:
            logger.warning(f"{src} not found; metadata written without copying the model")

        run_id = None
        results_json = self.results_csv.with_suffix(".json")
        if results_json.exists():
            run_id = json.loads(results_json.read_text(encoding="utf-8")).get("run_id")

        metadata = {
            "best_model_name": best,
            "tier_1_fast_fallback": best,  # read by PredictionPipeline: the model it serves
            "metrics": {k: (None if pd.isna(v) else v) for k, v in best_row.to_dict().items()},
            "selected_from": str(self.results_csv),
            "run_id": run_id,
            "excluded_baselines": sorted(BASELINES),
        }
        save_json(path=Path(self.config.metadata_path), data=metadata)
        return best
