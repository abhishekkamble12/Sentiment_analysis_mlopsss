"""Per-term explanations for the served linear text model (TF-IDF + linear classifier, optionally calibrated).

For a TF-IDF vector x the score of class c is W[c] @ x + b[c], so term j contributes x[j] * W[c, j]: exact
for a plain linear model. For CalibratedClassifierCV(sigmoid) each fold maps its margin f to
sigmoid(a * f + b) with a < 0, so -a * coef is that fold's coefficient on the calibrated scale; the folds are
averaged. That is an approximation (the model averages probabilities, not scores), which is why the
contributions are labelled approximate and only their ranking is shown to users.
"""
import re
from typing import Any, Dict, List, Optional

import numpy as np

SENTIMENT_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


def _linear_weights(model) -> Optional[tuple]:
    """(classes, W, b, exact) or None when the model is not linear with one row per class."""
    classes = np.asarray(getattr(model, "classes_", []))
    folds = getattr(model, "calibrated_classifiers_", None)
    parts = []
    for fold in (folds if folds is not None else [None]):
        est = model if fold is None else fold.estimator
        coef = getattr(est, "coef_", None)
        if coef is None or coef.shape[0] != len(classes):
            return None  # non-linear, or binary one-row coef_
        intercept = np.broadcast_to(getattr(est, "intercept_", 0.0), (len(classes),))
        scale = np.ones(len(classes))
        if fold is not None:  # sigmoid calibrators have a slope a_ (< 0); isotonic is monotone, keep 1
            scale = np.array([-getattr(cal, "a_", -1.0) for cal in fold.calibrators])
        parts.append((np.asarray(coef) * scale[:, None], intercept * scale))
    W = np.mean([p[0] for p in parts], axis=0)
    b = np.mean([p[1] for p in parts], axis=0)
    return classes, W, b, folds is None


def _term_pattern(term: str) -> re.Pattern:
    # clean_text lowercases and deletes punctuation, so "don't" became "dont": allow punctuation inside a
    # token and any non-word run between tokens of a bigram.
    tokens = [r"[^\w\s]*".join(map(re.escape, tok)) for tok in term.split()]
    return re.compile(r"(?<!\w)" + r"\W+".join(tokens) + r"(?!\w)", re.IGNORECASE)


def find_spans(text: str, term: str) -> List[List[int]]:
    """[start, end) character spans of a vectorizer term in the original (uncleaned) text."""
    return [[m.start(), m.end()] for m in _term_pattern(term).finditer(text)]


class LinearExplainer:
    def __init__(self, model, vectorizer):
        weights = _linear_weights(model)
        if weights is None:
            raise ValueError(f"{type(model).__name__} is not a linear model with per-class coefficients")
        self.classes, self.W, self.b, self.exact = weights
        self.vectorizer = vectorizer
        self.feature_names = np.asarray(vectorizer.get_feature_names_out())

    @classmethod
    def maybe(cls, model, vectorizer) -> Optional["LinearExplainer"]:
        """An explainer, or None when the model cannot be explained this way (never a fake explanation)."""
        try:
            return cls(model, vectorizer)
        except Exception:
            return None

    def contributions(self, x_row, label: int) -> tuple:
        """(feature indices, contributions to class `label`) for the non-zero terms of a 1-row sparse x."""
        c = int(np.flatnonzero(self.classes == label)[0])
        x = x_row.tocsr()
        return x.indices, x.data * self.W[c, x.indices]

    def explain(self, text: str, cleaned_text: str, label: int, k: int = 5) -> Dict[str, Any]:
        idx, contrib = self.contributions(self.vectorizer.transform([cleaned_text]), label)
        order = np.argsort(-np.abs(contrib))

        def item(i):
            term = str(self.feature_names[idx[i]])
            return {"term": term, "weight": round(float(contrib[i]), 4), "spans": find_spans(text, term)}

        return {
            "method": "linear_contributions" if self.exact else "linear_contributions_approx",
            "label": SENTIMENT_NAMES.get(label, str(label)),
            "supporting": [item(i) for i in order if contrib[i] > 0][:k],
            "opposing": [item(i) for i in order if contrib[i] < 0][:k],
            "note": "Term weights toward the predicted class." + (
                "" if self.exact else " Approximate for a calibrated model: read the ranking, not the values."),
        }
