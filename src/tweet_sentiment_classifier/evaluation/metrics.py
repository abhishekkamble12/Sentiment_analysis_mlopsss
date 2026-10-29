"""One evaluation routine for every model (silver or gold labels), plus CIs, calibration and latency.

Pure numpy: no MLflow, no plotting. Everything is computed from the confusion matrix so that
bootstrap resampling over 30k+ rows takes about a second.
"""
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

CLASSES = (-1, 0, 1)
CLASS_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


def _encode(y: Iterable, classes: Sequence[int] = CLASSES) -> np.ndarray:
    """Labels (ints, floats like -1.0, or numeric strings) -> class indices 0..k-1."""
    arr = np.asarray(list(y) if not isinstance(y, np.ndarray) else y).astype(float)
    ints = arr.astype(int)
    if not np.array_equal(arr, ints):
        raise ValueError("labels must be integers")
    classes_arr = np.asarray(classes)
    idx = np.searchsorted(classes_arr, ints)
    idx = np.clip(idx, 0, len(classes_arr) - 1)
    if not np.array_equal(classes_arr[idx], ints):
        raise ValueError(f"labels outside {tuple(classes)}: {sorted(set(ints.tolist()) - set(classes))}")
    return idx


def _confusion(t: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(t * k + p, minlength=k * k).reshape(k, k)


def _scores(cm: np.ndarray) -> Dict[str, Any]:
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    row, col = cm.sum(1).astype(float), cm.sum(0).astype(float)
    precision = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    recall = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    po = tp.sum() / n if n else 0.0
    pe = float((row * col).sum() / (n * n)) if n else 0.0
    kappa = (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else 0.0)
    return {
        "accuracy": float(po),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * row).sum() / n) if n else 0.0,
        "kappa": float(kappa),
        "precision": precision, "recall": recall, "f1": f1, "support": row.astype(int),
    }


def expected_calibration_error(y_true: Iterable, proba: np.ndarray, classes: Sequence[int] = CLASSES,
                               n_bins: int = 10) -> float:
    """Top-label ECE: weighted gap between confidence and accuracy over equal-width confidence bins."""
    proba = np.asarray(proba, dtype=float)
    t = _encode(y_true, classes)
    conf, pred = proba.max(1), proba.argmax(1)
    correct = (pred == t).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def evaluate(y_true: Iterable, y_pred: Iterable, proba: Optional[np.ndarray] = None,
             classes: Sequence[int] = CLASSES) -> Dict[str, Any]:
    t, p = _encode(y_true, classes), _encode(y_pred, classes)
    if len(t) != len(p):
        raise ValueError(f"y_true has {len(t)} rows but y_pred has {len(p)}")
    k = len(classes)
    cm = _confusion(t, p, k)
    s = _scores(cm)
    result: Dict[str, Any] = {
        "n": int(len(t)),
        "accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "weighted_f1": s["weighted_f1"], "kappa": s["kappa"],
        "per_class": {
            CLASS_NAMES.get(c, str(c)): {"precision": float(s["precision"][i]), "recall": float(s["recall"][i]),
                                         "f1": float(s["f1"][i]), "support": int(s["support"][i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
    }
    if proba is not None:
        result["ece"] = expected_calibration_error(y_true, proba, classes)
    return result


def bootstrap_ci(y_true: Iterable, y_pred: Iterable, metric: str = "macro_f1", n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05, classes: Sequence[int] = CLASSES) -> Tuple[float, float]:
    """Percentile bootstrap CI over test rows (resampling rows, models fixed)."""
    if metric not in ("accuracy", "macro_f1", "weighted_f1", "kappa"):
        raise ValueError(f"unsupported metric {metric!r}")
    t, p = _encode(y_true, classes), _encode(y_pred, classes)
    k, n = len(classes), len(t)
    code = t * k + p
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        cm = np.bincount(code[rng.integers(0, n, n)], minlength=k * k).reshape(k, k)
        values[b] = _scores(cm)[metric]
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def measure_latency(predict_one: Callable[[str], Any], texts: Sequence[str], n: int = 200,
                    warmup: int = 5) -> Dict[str, float]:
    """Per-text latency of single-item prediction (the serving pattern), excluding model loading."""
    sample = list(texts)[:n]
    for text in sample[:warmup]:
        predict_one(text)
    times = []
    for text in sample:
        start = time.perf_counter()
        predict_one(text)
        times.append((time.perf_counter() - start) * 1000)
    arr = np.asarray(times)
    return {"n": len(arr), "p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean())}


# ---- reproducible run output ---------------------------------------------------------------------

def fingerprint(texts: Iterable[str]) -> str:
    """Hash of the exact evaluation texts, so two result tables can be proven comparable."""
    h = hashlib.sha256()
    for text in texts:
        h.update(str(text).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def save_run(out_dir: Path, run_id: str, split_info: Dict[str, Any], results: Dict[str, Dict[str, Any]],
             comparisons: Optional[list] = None) -> None:
    """Write results.json and results.csv from scratch for one run (never appends to an old table)."""
    sizes = {name: r["n"] for name, r in results.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(f"models were evaluated on different numbers of rows: {sizes}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "split": split_info, "results": results, "comparisons": comparisons or []}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    comp_path = out_dir / "comparisons.csv"
    if comparisons:
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    elif comp_path.exists():
        comp_path.unlink()  # never leave a stale table from an earlier run next to new results

    columns = ["model", "n", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "weighted_f1", "kappa", "ece",
               "f1_negative", "f1_neutral", "f1_positive", "latency_p50_ms", "latency_p95_ms"]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for name, r in results.items():
            ci = r.get("macro_f1_ci") or (None, None)
            lat = r.get("latency") or {}
            pc = r["per_class"]
            writer.writerow([name, r["n"], r["accuracy"], r["macro_f1"], ci[0], ci[1], r["weighted_f1"], r["kappa"],
                             r.get("ece"), pc["negative"]["f1"], pc["neutral"]["f1"], pc["positive"]["f1"],
                             lat.get("p50_ms"), lat.get("p95_ms")])


# ---- comparing two models on the same rows ---------------------------------------------------------------

def paired_bootstrap(y_true: Iterable, pred_a: Iterable, pred_b: Iterable, metric: str = "macro_f1",
                     n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                     classes: Sequence[int] = CLASSES) -> Dict[str, Any]:
    """Difference metric(A) - metric(B) with a percentile CI, resampling the SAME rows for both models.

    p_value is the two-sided share of resamples falling on the other side of zero (floored at
    1/(n_boot+1)); it is an approximate p-value, adequate for ranking differences, not for tiny alphas.
    """
    if metric not in ("accuracy", "macro_f1", "weighted_f1", "kappa"):
        raise ValueError(f"unsupported metric {metric!r}")
    t, a, b = _encode(y_true, classes), _encode(pred_a, classes), _encode(pred_b, classes)
    k, n = len(classes), len(t)
    code_a, code_b = t * k + a, t * k + b

    def score(code):
        return _scores(np.bincount(code, minlength=k * k).reshape(k, k))[metric]

    point = score(code_a) - score(code_b)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = score(code_a[idx]) - score(code_b[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {"diff": float(point), "ci": (float(lo), float(hi)), "p_value": float(min(1.0, max(p, 1 / (n_boot + 1))))}


def mcnemar_exact(y_true: Iterable, pred_a: Iterable, pred_b: Iterable,
                  classes: Sequence[int] = CLASSES) -> Dict[str, Any]:
    """Exact McNemar test on the rows where exactly one of the two models is correct."""
    from scipy.stats import binomtest
    t, a, b = _encode(y_true, classes), _encode(pred_a, classes), _encode(pred_b, classes)
    only_a = int(((a == t) & (b != t)).sum())
    only_b = int(((a != t) & (b == t)).sum())
    n = only_a + only_b
    p = float(binomtest(min(only_a, only_b), n, 0.5).pvalue) if n else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def holm_adjust(p_values: Sequence[float]) -> list:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted
