import sys
import os
import importlib.util
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.config.configuration import ConfigurationManager
from tweet_sentiment_classifier.components import (
    DataIngestion,
    DataValidation,
    DataTransformation,
    BestModelSelector
)
from tweet_sentiment_classifier.evaluation import run as evaluation
from tweet_sentiment_classifier.models.export import export_models
from tweet_sentiment_classifier.models.factory import SKLEARN_MODELS


def _available_baselines():
    """Majority always; TextBlob / VADER when requirements/eval.txt is installed."""
    baselines = ["Majority"]
    for name, module in (("TextBlob", "textblob"), ("VADER", "vaderSentiment")):
        if importlib.util.find_spec(module):
            baselines.append(name)
        else:
            logger.warning(f"{name} baseline skipped: install requirements/eval.txt")
    return baselines


def _log_to_mlflow(summary):
    """Log the single evaluation run to MLflow when it is installed (requirements/train.txt)."""
    if not importlib.util.find_spec("mlflow"):
        logger.info("MLflow not installed; skipping tracking")
        return
    import mlflow
    mlflow.set_experiment("Tweet_Sentiment_Classification")
    with mlflow.start_run(run_name=summary["run_id"]):
        mlflow.set_tags({"test_fingerprint": summary["split"]["test_fingerprint"],
                         "label_source": summary["split"]["label_source"]})
        for name, r in summary["results"].items():
            with mlflow.start_run(run_name=name, nested=True):
                metrics = {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "kappa": r["kappa"],
                           "macro_f1_ci_low": r["macro_f1_ci"][0], "macro_f1_ci_high": r["macro_f1_ci"][1]}
                if "ece" in r:
                    metrics["ece"] = r["ece"]
                if "latency" in r:
                    metrics["latency_p50_ms"] = r["latency"]["p50_ms"]
                mlflow.log_metrics(metrics)
        mlflow.log_artifacts("artifacts/evaluation", artifact_path="evaluation")


try:
    config_manager = ConfigurationManager()
    params = config_manager.params

    logger.info(">>>>>> Stage 1: Data Ingestion <<<<<<")
    data_ingestion = DataIngestion(config=config_manager.get_data_ingestion_config())
    local_data_path = data_ingestion.download_file()
    data_ingestion.extract_zip_file(local_data_path)

    logger.info(">>>>>> Stage 2: Data Validation <<<<<<")
    if not DataValidation(config=config_manager.get_data_validation_config()).validate_all_data():
        raise Exception("Data Validation Failed. Check status file or logs.")

    logger.info(">>>>>> Stage 3: Clean, de-duplicate, stratified train/val/test split <<<<<<")
    transformation_config = config_manager.get_data_transformation_config()
    DataTransformation(config=transformation_config).transform_and_split()

    logger.info(">>>>>> Stage 4: Train and export the served models <<<<<<")
    requested = list(params.get("models_to_train", ["LogisticRegression"]))
    trained = [m for m in requested if m in SKLEARN_MODELS]
    for skipped in sorted(set(requested) - set(trained)):
        logger.warning(f"{skipped} is not trained by this pipeline (transformers: models/finetune.py)")
    max_features = int(params.get("max_features", 15000))
    seed = int(params.get("random_state", 42))
    export_models(transformation_config.train_data_path, trained, params, seed=seed, max_features=max_features)

    logger.info(">>>>>> Stage 5: Evaluate every model in one run <<<<<<")
    summary = evaluation.run(
        Path(transformation_config.train_data_path), Path(transformation_config.test_data_path),
        Path("artifacts/evaluation"), models=_available_baselines() + trained,
        params=params, seed=seed, max_features=max_features,
    )
    logger.info("\n" + evaluation._format(summary))

    logger.info(">>>>>> Stage 6: Select the served model <<<<<<")
    best = BestModelSelector(config=config_manager.get_best_model_config()).select_best_model()
    logger.info(f"Serving {best}")

    _log_to_mlflow(summary)
    logger.info(">>>>>> Pipeline completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e
