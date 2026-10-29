"""Assemble paper tables and figures from result files (numbers are never typed by hand).

    python -m tweet_sentiment_classifier.experiments.report --silver artifacts/evaluation/results.json \
        --gold artifacts/evaluation/gold/results.json --budget artifacts/experiments/gold_budget/results.json \
        --out paper

    python -m tweet_sentiment_classifier.experiments.report errors --predictions artifacts/evaluation/gold/predictions.csv \
        --model LinearSVC --n 200 --out artifacts/experiments/error_sample.csv
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def silver_vs_gold_table(silver: Dict, gold: Optional[Dict]) -> pd.DataFrame:
    rows = []
    for name, r in silver["results"].items():
        row = {"model": name, "silver_macro_f1": r["macro_f1"], "silver_ci_low": r["macro_f1_ci"][0],
               "silver_ci_high": r["macro_f1_ci"][1]}
        g = (gold or {}).get("results", {}).get(name)
        if g:
            row.update(gold_macro_f1=g["macro_f1"], gold_ci_low=g["macro_f1_ci"][0], gold_ci_high=g["macro_f1_ci"][1],
                       gap=r["macro_f1"] - g["macro_f1"])
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(table: pd.DataFrame) -> str:
    def cell(v, lo=None, hi=None):
        if v is None or pd.isna(v):
            return "--"
        return f"{v:.3f}" if lo is None else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Model & Silver macro-F1 & Gold macro-F1 & Gap \\", r"\midrule"]
    for _, r in table.iterrows():
        gold = cell(r.get("gold_macro_f1"), r.get("gold_ci_low"), r.get("gold_ci_high")) if "gold_macro_f1" in r else "--"
        gap = cell(r.get("gap")) if "gap" in r else "--"
        lines.append(f"{r['model']} & {cell(r['silver_macro_f1'], r['silver_ci_low'], r['silver_ci_high'])} & {gold} & {gap} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def budget_figure(budget: Dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = pd.DataFrame(budget["summary"])
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for col, label, color in (("silver_plus_gold", "silver + k gold", "#2563eb"), ("gold_only", "k gold only", "#d97706"),
                              ("silver_only", "silver only", "#6b7280")):
        if f"{col}_mean" in s:
            ax.errorbar(s["k"], s[f"{col}_mean"], yerr=s.get(f"{col}_std"), label=label, color=color, marker="o", capsize=3)
    ax.set_xlabel("human-labelled examples (k)")
    ax.set_ylabel("gold_test macro-F1")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def error_sample(predictions_csv: Path, model: str, n: int, seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    col = f"pred_{model}"
    if col not in df:
        raise ValueError(f"{predictions_csv} has no column {col}")
    errors = df[df[col] != df["label"]]
    sample = errors.sample(n=min(n, len(errors)), random_state=seed)
    return sample[["text", "label", col]].assign(category="", notes="")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="tables", choices=["tables", "errors"])
    parser.add_argument("--silver", type=Path, default=Path("artifacts/evaluation/results.json"))
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--budget", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--model", default="LinearSVC")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("paper"))
    args = parser.parse_args(argv)

    if args.mode == "errors":
        sample = error_sample(args.predictions, args.model, args.n)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(args.out, index=False)
        print(f"wrote {len(sample)} errors to {args.out}; a human fills `category`")
        return

    silver = json.loads(args.silver.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8")) if args.gold and args.gold.exists() else None
    table = silver_vs_gold_table(silver, gold)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "tables" / "table1_silver_vs_gold.csv", index=False)
    (args.out / "tables" / "table1_silver_vs_gold.tex").write_text(to_latex(table), encoding="utf-8")
    if args.budget and args.budget.exists():
        budget_figure(json.loads(args.budget.read_text(encoding="utf-8")), args.out / "figures" / "gold_budget.png")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
