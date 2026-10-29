"""How much do the results move when only the random split changes?

    python -m tweet_sentiment_classifier.evaluation.seeds --data sentiment_analysis/dataset/thoroughly_cleaned_data.csv

For each seed this re-runs the full clean -> de-duplicate -> stratified split on the source file, fits each
model from scratch and scores it on that seed's test set. Reporting mean and standard deviation over seeds
shows whether a gap between two models is bigger than split-to-split noise.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from tweet_sentiment_classifier.components.data_transformation import clean_and_split
from tweet_sentiment_classifier.evaluation.metrics import evaluate
from tweet_sentiment_classifier.evaluation.run import TfidfModel

METRICS = ("macro_f1", "accuracy", "kappa")
DEFAULT_MODELS = ("LogisticRegression", "LinearSVC", "MultinomialNB", "RidgeClassifier")


def run_seed_study(data_csv: Path, seeds: Sequence[int], models: Sequence[str] = DEFAULT_MODELS,
                   params: Dict[str, Any] = None, test_size: float = 0.20, val_size: float = 0.10,
                   max_features: int = 15000, progress=print) -> Dict[str, Any]:
    params = params or {}
    raw = pd.read_csv(data_csv)
    per_seed: Dict[int, Dict[str, Any]] = {}
    values: Dict[str, Dict[str, list]] = {m: {k: [] for k in METRICS} for m in models}

    for seed in seeds:
        train, _, test, manifest = clean_and_split(raw, test_size, val_size, seed)
        x_tr, y_tr = train["text"].astype(str).tolist(), train["sentiment"].astype(int).to_numpy()
        x_te, y_te = test["text"].astype(str).tolist(), test["sentiment"].astype(int).to_numpy()
        per_seed[seed] = {"test_fingerprint": manifest["splits"]["test"]["fingerprint"], "n_test": len(x_te),
                          "exact_overlap": manifest["exact_text_overlap"]["train_test"]}
        for name in models:
            model = TfidfModel(name, params.get(name, {}), seed, max_features).fit(x_tr, y_tr)
            scores = evaluate(y_te, model.predict(x_te))
            for k in METRICS:
                values[name][k].append(scores[k])
            progress(f"seed {seed:<4}{name:<19}macroF1={scores['macro_f1']:.4f} acc={scores['accuracy']:.4f}")

    summary = {
        name: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                   "min": float(np.min(v)), "max": float(np.max(v)), "values": [float(x) for x in v]}
               for k, v in metrics.items()}
        for name, metrics in values.items()
    }
    return {"seeds": list(seeds), "test_size": test_size, "val_size": val_size, "per_seed": per_seed,
            "summary": summary}


def save_study(study: Dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seed_study.json").write_text(json.dumps(study, indent=2), encoding="utf-8")
    with open(out_dir / "seed_study.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n_seeds", "macro_f1_mean", "macro_f1_sd", "macro_f1_min", "macro_f1_max",
                         "accuracy_mean", "accuracy_sd", "kappa_mean"])
        for name, s in study["summary"].items():
            writer.writerow([name, len(study["seeds"]), s["macro_f1"]["mean"], s["macro_f1"]["sd"], s["macro_f1"]["min"],
                             s["macro_f1"]["max"], s["accuracy"]["mean"], s["accuracy"]["sd"], s["kappa"]["mean"]])


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation"))
    args = parser.parse_args(argv)

    import yaml
    params = yaml.safe_load(args.params.read_text(encoding="utf-8")) if args.params.exists() else {}
    study = run_seed_study(args.data, args.seeds, args.models, params, max_features=int(params.get("max_features", 15000)))
    save_study(study, args.out)
    print(f"\n{'model':<20}{'macro-F1 mean':>14}{'sd':>9}{'min':>9}{'max':>9}")
    for name, s in study["summary"].items():
        m = s["macro_f1"]
        print(f"{name:<20}{m['mean']:>14.4f}{m['sd']:>9.4f}{m['min']:>9.4f}{m['max']:>9.4f}")
    print(f"\nwrote {args.out / 'seed_study.csv'} and seed_study.json")


if __name__ == "__main__":
    main()
