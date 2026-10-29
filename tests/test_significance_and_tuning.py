"""Paired tests, tuning on validation only, and the multi-seed study."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from tweet_sentiment_classifier.evaluation import run as runner
from tweet_sentiment_classifier.evaluation import seeds as seeds_mod
from tweet_sentiment_classifier.evaluation import tune as tune_mod
from tweet_sentiment_classifier.evaluation.metrics import holm_adjust, mcnemar_exact, paired_bootstrap, save_run, evaluate


def labels(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 0, 1], size=n, p=[0.2, 0.35, 0.45])


def corrupt(y, rate, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(len(y)) < rate, rng.choice([-1, 0, 1], size=len(y)), y)


# ---- paired bootstrap ----------------------------------------------------------------------------

def test_clearly_better_model_gets_a_positive_difference_and_small_p():
    y = labels()
    good, bad = corrupt(y, 0.1, 1), corrupt(y, 0.5, 2)
    r = paired_bootstrap(y, good, bad, "macro_f1", n_boot=300, seed=0)
    assert r["diff"] > 0.2 and r["ci"][0] > 0 and r["p_value"] < 0.01
    reverse = paired_bootstrap(y, bad, good, "macro_f1", n_boot=300, seed=0)
    assert reverse["diff"] == pytest.approx(-r["diff"]) and reverse["ci"][1] < 0


def test_identical_models_have_zero_difference_and_p_one():
    y = labels()
    p = corrupt(y, 0.3, 1)
    r = paired_bootstrap(y, p, p, "accuracy", n_boot=200, seed=0)
    assert r["diff"] == 0.0 and r["ci"] == (0.0, 0.0) and r["p_value"] == 1.0


def test_two_equally_good_models_are_not_declared_different():
    y = labels(3000)
    a, b = corrupt(y, 0.3, 1), corrupt(y, 0.3, 2)       # same error rate, independent mistakes
    r = paired_bootstrap(y, a, b, "accuracy", n_boot=400, seed=0)
    assert r["ci"][0] < 0 < r["ci"][1] and r["p_value"] > 0.05


def test_paired_bootstrap_is_seeded_and_rejects_bad_metric():
    y = labels()
    a, b = corrupt(y, 0.2, 1), corrupt(y, 0.3, 2)
    assert paired_bootstrap(y, a, b, n_boot=100, seed=7) == paired_bootstrap(y, a, b, n_boot=100, seed=7)
    with pytest.raises(ValueError):
        paired_bootstrap(y, a, b, "auc")


# ---- McNemar / Holm ---------------------------------------------------------------------------------

def test_mcnemar_counts_and_matches_the_exact_binomial():
    y = np.array([0] * 20)
    a = np.array([0] * 15 + [1] * 5)                     # A right on 15
    b = np.array([1] * 12 + [0] * 8)                     # B right on 8; both right on 3; only A on 12; only B on 5
    r = mcnemar_exact(y, a, b)
    assert (r["only_a_correct"], r["only_b_correct"]) == (12, 5)
    assert r["p_value"] == pytest.approx(binomtest(5, 17, 0.5).pvalue)


def test_mcnemar_with_no_disagreements_gives_p_one():
    y = labels(50)
    assert mcnemar_exact(y, y, y)["p_value"] == 1.0


def test_holm_adjustment_known_values():
    # classic example: m=4, sorted p = .01,.02,.03,.04 -> .04,.06,.06,.06
    assert holm_adjust([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []
    assert all(0 <= p <= 1 for p in holm_adjust([1e-9, 0.2, 0.6]))


# ---- tuning --------------------------------------------------------------------------------------

def test_tune_has_no_way_to_read_the_test_set():
    names = set(inspect.signature(tune_mod.tune).parameters)
    assert "test_csv" not in names and not any("test" in n for n in names)


def write_split(path, texts, y):
    pd.DataFrame({"text": texts, "sentiment": [float(v) for v in y]}).to_csv(path, index=False)


def toy(seed, n=240):
    rng = np.random.default_rng(seed)
    vocab = {-1: "awful corrupt failure angry hate".split(), 0: "report meeting today city date".split(),
             1: "great happy success proud love".split()}
    shared = "the modi vote people india govt party news time year".split()
    ys = np.array([(-1, 0, 1)[i % 3] for i in range(n)])
    # 1 class word among 4 shared words: separable, but only if the model is not over-regularised
    texts = [" ".join(list(rng.choice(vocab[y], 1)) + list(rng.choice(shared, 4))) for y in ys]
    return texts, ys


def test_tune_picks_the_best_validation_setting_and_records_every_trial(tmp_path):
    write_split(tmp_path / "train.csv", *toy(1))
    write_split(tmp_path / "val.csv", *toy(2, 90))
    grids = {"MultinomialNB": [{"alpha": 1000.0}, {"alpha": 0.1}],       # absurdly smooth NB must lose
             "LogisticRegression": [{"C": 0.0001, "max_iter": 200}, {"C": 10.0, "max_iter": 200}]}
    res = tune_mod.tune(tmp_path / "train.csv", tmp_path / "val.csv", tmp_path / "t.json", grids, progress=lambda *_: None)
    assert res["best"]["MultinomialNB"]["params"] == {"alpha": 0.1}
    assert res["best"]["LogisticRegression"]["params"]["C"] == 10.0
    assert len(res["trials"]["MultinomialNB"]) == 2 and "converged" in res["trials"]["LogisticRegression"][0]
    assert json.loads((tmp_path / "t.json").read_text())["note"].startswith("chosen on the validation split only")


def test_tie_keeps_the_earlier_simpler_setting(tmp_path):
    write_split(tmp_path / "train.csv", *toy(1))
    write_split(tmp_path / "val.csv", *toy(2, 90))
    same = [{"alpha": 0.1}, {"alpha": 0.1}]                              # identical results
    res = tune_mod.tune(tmp_path / "train.csv", tmp_path / "val.csv", tmp_path / "t.json",
                        {"MultinomialNB": same}, progress=lambda *_: None)
    assert res["best"]["MultinomialNB"]["params"] == {"alpha": 0.1}


# ---- comparisons wired into the runner ----------------------------------------------------------------------

def test_runner_writes_holm_adjusted_comparisons_and_removes_stale_ones(tmp_path):
    tr_x, tr_y = toy(1, 300)
    te_x, te_y = toy(2, 150)
    te_x = [t + " extra" for t in te_x]
    write_split(tmp_path / "train.csv", tr_x, tr_y)
    write_split(tmp_path / "test.csv", te_x, te_y)
    out = tmp_path / "out"
    summary = runner.run(tmp_path / "train.csv", tmp_path / "test.csv", out,
                         models=("Majority", "LogisticRegression", "MultinomialNB", "RidgeClassifier"), n_boot=100)
    comps = pd.read_csv(out / "comparisons.csv")
    assert len(comps) == 3 and "Majority" not in set(comps["model_a"]) | set(comps["model_b"])
    assert (comps["p_bootstrap_holm"] >= comps["p_bootstrap"]).all() and (comps["mcnemar_p_holm"] >= comps["mcnemar_p"]).all()
    assert len(summary["comparisons"]) == 3
    runner.run(tmp_path / "train.csv", tmp_path / "test.csv", out, models=("Majority", "LogisticRegression"), n_boot=50)
    assert not (out / "comparisons.csv").exists()                         # one real model -> no pairs -> no stale table


def test_save_run_without_comparisons_writes_empty_list(tmp_path):
    m = evaluate([0, 1], [0, 1])
    m["macro_f1_ci"] = (0.0, 1.0)
    save_run(tmp_path, "r", {}, {"A": m})
    assert json.loads((tmp_path / "results.json").read_text())["comparisons"] == []


# ---- seed study ---------------------------------------------------------------------------------------------------

def test_seed_study_reruns_the_whole_split_per_seed(tmp_path):
    texts, ys = toy(5, 600)
    write_split(tmp_path / "all.csv", texts, ys)
    study = seeds_mod.run_seed_study(tmp_path / "all.csv", [1, 2, 3], models=("MultinomialNB", "LogisticRegression"),
                                     progress=lambda *_: None)
    fps = [study["per_seed"][s]["test_fingerprint"] for s in (1, 2, 3)]
    assert len(set(fps)) == 3                                            # genuinely different test sets
    assert all(study["per_seed"][s]["exact_overlap"] == 0 for s in (1, 2, 3))
    nb = study["summary"]["MultinomialNB"]["macro_f1"]
    assert len(nb["values"]) == 3 and nb["min"] <= nb["mean"] <= nb["max"] and nb["sd"] >= 0
    seeds_mod.save_study(study, tmp_path / "o")
    assert list(pd.read_csv(tmp_path / "o" / "seed_study.csv")["model"]) == ["MultinomialNB", "LogisticRegression"]
