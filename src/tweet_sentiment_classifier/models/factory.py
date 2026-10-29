"""Single place that builds the TF-IDF-feature classifiers, used by both training and evaluation."""
from typing import Any, Dict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

SKLEARN_MODELS = ("LogisticRegression", "LinearSVC", "MultinomialNB", "RidgeClassifier", "XGBoost")


def build_classifier(model_name: str, hyperparams: Dict[str, Any], random_state: int):
    """Unfitted classifier. Margin models are calibrated so every model exposes predict_proba."""
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=hyperparams.get("C", 1.0),
            solver=hyperparams.get("solver", "lbfgs"),
            max_iter=hyperparams.get("max_iter", 100),
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "LinearSVC":
        return CalibratedClassifierCV(
            LinearSVC(
                C=hyperparams.get("C", 1.0),
                loss=hyperparams.get("loss", "squared_hinge"),
                max_iter=hyperparams.get("max_iter", 1000),
                random_state=random_state,
            ),
            cv=3,
        )
    if model_name == "MultinomialNB":
        return MultinomialNB(alpha=hyperparams.get("alpha", 0.1))
    if model_name == "RidgeClassifier":
        return CalibratedClassifierCV(
            RidgeClassifier(alpha=hyperparams.get("alpha", 1.0), random_state=random_state),
            cv=3,
        )
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=hyperparams.get("n_estimators", 80),
            max_depth=hyperparams.get("max_depth", 5),
            learning_rate=hyperparams.get("learning_rate", 0.15),
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    raise ValueError(f"Unknown classifier: {model_name}")
