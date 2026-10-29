"""How many human labels does it take to fix a lexicon-trained model?

    python -m tweet_sentiment_classifier.experiments.gold_budget --pool data/gold/gold_train_pool.csv \
        --gold-test data/gold/gold_test.csv --out artifacts/experiments/gold_budget

For each budget k and seed, k gold examples are drawn (stratified) from gold_train_pool and three
TF-IDF + LogisticRegression variants are scored on gold_test:
  silver_only        trained on the silver training split (independent of k)
  silver_plus_gold   silver + the k gold rows up-weighted by w; w in --weights chosen by 3-fold CV on the
                     k gold rows themselves (never on gold_test)
  gold_only          trained on the k gold rows alone
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from tweet_sentiment_classifier.evaluation.metrics import bootstrap_ci, evaluate
from tweet_sentiment_classifier.models.factory import build_classifier


def _fit(X, y, w, params, seed):
    return build_classifier("LogisticRegression", params, seed).fit(X, y, sample_weight=w)


def _choose_weight(Xs, ys, Xg, yg, weights, params, seed) -> float:
    """Pick the gold up-weight by cross-validation inside the k gold rows."""
    if len(yg) < 15 or min(np.bincount(yg + 1, minlength=3)) < 3:
        return float(weights[0])
    scores = {}
    for w in weights:
        fold_scores = []
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(yg)), yg):
            X = vstack([Xs, Xg[tr]])
            y = np.concatenate([ys, yg[tr]])
            sw = np.concatenate([np.ones(len(ys)), np.full(len(tr), w)])
            fold_scores.append(evaluate(yg[va], _fit(X, y, sw, params, seed).predict(Xg[va]))["macro_f1"])
        scores[w] = float(np.mean(fold_scores))
    return float(max(weights, key=lambda w: scores[w]))  # first of equal scores = smallest weight


def run_budget(silver_train: Path, pool_csv: Path, gold_test_csv: Path, out_dir: Path,
               budgets: Sequence[int] = (0, 50, 100, 200, 400), seeds: Sequence[int] = (0, 1, 2, 3, 4),
               weights: Sequence[float] = (1.0, 10.0, 50.0), params: Dict = None, max_features: int = 15000,
               silver_fraction: float = 1.0, n_boot: int = 500, progress=print) -> Dict:
    params = params or {"C": 4.0, "max_iter": 1000}
    silver = pd.read_csv(silver_train).dropna(subset=["text"])
    if silver_fraction < 1.0:
        silver = silver.groupby("sentiment", group_keys=False).sample(frac=silver_fraction, random_state=0)
    pool = pd.read_csv(pool_csv).dropna(subset=["text"]).reset_index(drop=True)
    test = pd.read_csv(gold_test_csv).dropna(subset=["text"])

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)).fit(silver["text"].astype(str))
    Xs, ys = vec.transform(silver["text"].astype(str)), silver["sentiment"].astype(int).to_numpy()
    Xt, yt = vec.transform(test["text"].astype(str)), test["sentiment"].astype(int).to_numpy()
    Xp, yp = vec.transform(pool["text"].astype(str)), pool["sentiment"].astype(int).to_numpy()

    def score(model):
        pred = model.predict(Xt)
        return {"macro_f1": evaluate(yt, pred)["macro_f1"], "macro_f1_ci": bootstrap_ci(yt, pred, n_boot=n_boot)}

    silver_only = score(_fit(Xs, ys, None, params, 0))
    rows = []
    for k in budgets:
        if k > len(pool):
            raise ValueError(f"budget {k} exceeds the gold pool ({len(pool)})")
        for seed in seeds:
            row = {"k": k, "seed": seed, "silver_only": silver_only["macro_f1"]}
            if k == 0:
                row.update(silver_plus_gold=silver_only["macro_f1"], gold_only=None, chosen_weight=None)
            else:
                pick = (pool.groupby("sentiment", group_keys=False)
                        .sample(frac=k / len(pool), random_state=seed).index[:k])
                Xg, yg = Xp[pick], yp[pick]
                w = _choose_weight(Xs, ys, Xg, yg, weights, params, seed)
                X = vstack([Xs, Xg])
                y = np.concatenate([ys, yg])
                sw = np.concatenate([np.ones(len(ys)), np.full(len(yg), w)])
                row["chosen_weight"] = w
                row["silver_plus_gold"] = score(_fit(X, y, sw, params, seed))["macro_f1"]
                row["gold_only"] = score(_fit(Xg, yg, None, params, seed))["macro_f1"] if len(set(yg)) == 3 else None
            rows.append(row)
            progress(f"k={k:<4} seed={seed} silver+gold={row['silver_plus_gold']:.4f}")

    table = pd.DataFrame(rows)
    summary = (table.groupby("k")[["silver_only", "silver_plus_gold", "gold_only"]]
               .agg(["mean", "std"]).reset_index())
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "runs.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    result = {"silver_only": silver_only, "budgets": list(budgets), "seeds": list(seeds), "weights": list(weights),
              "n_pool": len(pool), "n_gold_test": len(test), "summary": summary.to_dict(orient="records")}
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--silver-train", type=Path, default=Path("artifacts/data_transformation/train.csv"))
    parser.add_argument("--pool", type=Path, default=Path("data/gold/gold_train_pool.csv"))
    parser.add_argument("--gold-test", type=Path, default=Path("data/gold/gold_test.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/experiments/gold_budget"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 50, 100, 200, 400])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--silver-fraction", type=float, default=1.0)
    args = parser.parse_args(argv)
    run_budget(args.silver_train, args.pool, args.gold_test, args.out, args.budgets, args.seeds,
               silver_fraction=args.silver_fraction)
    print(pd.read_csv(args.out / "summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
