"""Gold-set tooling: sampling, agreement, adjudication, freezing, weighted metrics. No real labels involved."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from tweet_sentiment_classifier.evaluation.metrics import bootstrap_ci, evaluate
from tweet_sentiment_classifier.gold.adjudicate import adjudicate, build_gold
from tweet_sentiment_classifier.gold.agreement import agreement_report, krippendorff_alpha_nominal, load_annotations
from tweet_sentiment_classifier.gold.freeze import freeze
from tweet_sentiment_classifier.gold.sample import draw_sample


def write_test_split(path, counts=(220, 340, 440)):
    rows = []
    for label, n in zip((-1, 0, 1), counts):
        rows += [{"text": f"tweet {label} number {i}", "sentiment": float(label)} for i in range(n)]
    pd.DataFrame(rows).to_csv(path, index=False)


# ---- sampling ---------------------------------------------------------------------------------

def test_sample_is_class_balanced_hides_silver_labels_and_weights_back_to_the_natural_mix(tmp_path):
    write_test_split(tmp_path / "test.csv")
    s = draw_sample(tmp_path / "test.csv", tmp_path / "g", n_per_class=50, seed=1)
    assert s["silver_label"].value_counts().to_dict() == {-1: 50, 0: 50, 1: 50}
    shown = pd.read_csv(tmp_path / "g" / "to_annotate.csv")
    assert list(shown.columns) == ["id", "text"]                                 # annotators never see labels
    # weights restore the natural shares 22% / 34% / 44%
    w = s.groupby("silver_label")["sampling_weight"].sum() / s["sampling_weight"].sum()
    assert w.to_dict() == pytest.approx({-1: 0.22, 0: 0.34, 1: 0.44})
    assert json.loads((tmp_path / "g" / "manifest.json").read_text())["n"] == 150


def test_main_sample_can_exclude_the_pilot(tmp_path):
    write_test_split(tmp_path / "test.csv")
    pilot = draw_sample(tmp_path / "test.csv", tmp_path / "g", 30, seed=1, prefix="pilot_")
    main = draw_sample(tmp_path / "test.csv", tmp_path / "g", 100, seed=2, exclude_private=tmp_path / "g" / "pilot_private.csv")
    assert not set(pilot["row_index"]) & set(main["row_index"])


def test_sample_refuses_when_a_class_is_too_small(tmp_path):
    write_test_split(tmp_path / "test.csv", counts=(10, 100, 100))
    with pytest.raises(ValueError, match="only 10"):
        draw_sample(tmp_path / "test.csv", tmp_path / "g", 50)


# ---- agreement ----------------------------------------------------------------------------------

def test_alpha_is_one_for_perfect_agreement_and_negative_for_systematic_disagreement():
    assert krippendorff_alpha_nominal({i: [i % 3 - 1] * 3 for i in range(30)}) == pytest.approx(1.0)
    assert krippendorff_alpha_nominal({i: [0, 1] if i % 2 else [1, 0] for i in range(40)}) < 0


def test_alpha_is_close_to_kappa_for_two_annotators_with_similar_marginals():
    rng = np.random.default_rng(0)
    a = rng.choice([-1, 0, 1], size=2000, p=[0.2, 0.35, 0.45])
    b = np.where(rng.random(2000) < 0.25, rng.choice([-1, 0, 1], size=2000, p=[0.2, 0.35, 0.45]), a)
    alpha = krippendorff_alpha_nominal({i: [int(x), int(y)] for i, (x, y) in enumerate(zip(a, b))})
    assert alpha == pytest.approx(cohen_kappa_score(a, b), abs=0.01)


def test_alpha_ignores_items_with_a_single_label():
    with_single = {0: [1, 1], 1: [0, 0], 2: [-1, -1], 3: [1]}
    assert krippendorff_alpha_nominal(with_single) == pytest.approx(1.0)


def ann(tmp_path, name, labels, flags=None):
    df = pd.DataFrame({"id": [f"g{i}" for i in range(len(labels))], "label": labels})
    if flags is not None:
        df["flags"] = flags
    p = tmp_path / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


def test_report_handles_names_skips_and_flags(tmp_path):
    a = ann(tmp_path, "a", ["positive", "negative", "neutral", "positive", ""], ["sarcasm", "", "", "", ""])
    b = ann(tmp_path, "b", [1, -1, 0, 0, 1], ["sarcasm;code_mixed", "", "", "", ""])
    report = agreement_report([load_annotations(a), load_annotations(b)], ["a", "b"])
    assert report["labels_per_annotator"] == {"a": 4, "b": 5}                   # blank = skipped
    assert report["pairwise"]["a|b"]["n_items"] == 4
    assert report["pairwise"]["a|b"]["percent_agreement"] == pytest.approx(0.75)
    assert report["flag_counts"]["sarcasm"] == {"a": 1, "b": 1} and report["flag_counts"]["code_mixed"]["b"] == 1


def test_unknown_label_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown label"):
        load_annotations(ann(tmp_path, "x", ["happy"]))


# ---- adjudication ---------------------------------------------------------------------------------

def test_adjudication_rules(tmp_path):
    a = load_annotations(ann(tmp_path, "a", [1, 1, 0, -1, 1]))
    b = load_annotations(ann(tmp_path, "b", [1, 0, -1, -1, ""]))
    c = load_annotations(ann(tmp_path, "c", ["", 0, 1, "", ""]))
    out = adjudicate(a, b, c).set_index("id")
    assert out.loc["g0", "status"] == "agreed" and out.loc["g0", "sentiment"] == 1
    assert out.loc["g1", "status"] == "adjudicated" and out.loc["g1", "sentiment"] == 0
    assert out.loc["g2", "status"] == "unresolved"                               # tie-break sides with neither
    assert out.loc["g3", "status"] == "agreed"
    assert out.loc["g4", "status"] == "single"


def test_build_gold_keeps_only_resolved_items_and_raw_files_are_untouched(tmp_path):
    a_path, b_path = ann(tmp_path, "a", [1, 0, 1]), ann(tmp_path, "b", [1, 0, -1])
    before = a_path.read_bytes()
    private = pd.DataFrame({"id": ["g0", "g1", "g2"], "row_index": [5, 6, 7], "text": ["x", "y", "z"],
                            "silver_label": [1, 0, 1], "sampling_weight": [1.0, 1.0, 1.0]})
    private.to_csv(tmp_path / "private.csv", index=False)
    gold = build_gold(a_path, b_path, tmp_path / "private.csv", tmp_path / "gold.csv")
    assert list(gold["id"]) == ["g0", "g1"] and gold["sentiment"].tolist() == [1, 0]
    assert a_path.read_bytes() == before
    assert len(pd.read_csv(tmp_path / "gold_all_statuses.csv")) == 3


# ---- freeze -------------------------------------------------------------------------------------------

def write_gold(path, n=1200):
    pd.DataFrame({"id": [f"g{i}" for i in range(n)], "text": [f"gold tweet {i}" for i in range(n)],
                  "sentiment": [(-1, 0, 1)[i % 3] for i in range(n)], "sampling_weight": 1.0}).to_csv(path, index=False)


def test_freeze_splits_and_fingerprints(tmp_path):
    write_gold(tmp_path / "gold.csv")
    pd.DataFrame({"text": ["other"], "sentiment": [1]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"text": ["another"], "sentiment": [0]}).to_csv(tmp_path / "val.csv", index=False)
    m = freeze(tmp_path / "gold.csv", tmp_path / "train.csv", tmp_path / "val.csv", tmp_path / "out", test_n=1000)
    test, pool = pd.read_csv(tmp_path / "out" / "gold_test.csv"), pd.read_csv(tmp_path / "out" / "gold_train_pool.csv")
    assert (len(test), len(pool)) == (1000, 200) and not set(test["id"]) & set(pool["id"])
    assert m["gold_test"]["n"] == 1000 and "fingerprint" in m["gold_train_pool"]


def test_freeze_refuses_gold_that_overlaps_training_data(tmp_path):
    write_gold(tmp_path / "gold.csv")
    pd.DataFrame({"text": ["gold tweet 7"], "sentiment": [1]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"text": ["x"], "sentiment": [0]}).to_csv(tmp_path / "val.csv", index=False)
    with pytest.raises(ValueError, match="also appear in train/val"):
        freeze(tmp_path / "gold.csv", tmp_path / "train.csv", tmp_path / "val.csv", tmp_path / "out")


def test_freeze_refuses_too_small_gold_set(tmp_path):
    write_gold(tmp_path / "gold.csv", n=500)
    pd.DataFrame({"text": ["x"], "sentiment": [0]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"text": ["y"], "sentiment": [0]}).to_csv(tmp_path / "val.csv", index=False)
    with pytest.raises(ValueError, match="only 500"):
        freeze(tmp_path / "gold.csv", tmp_path / "train.csv", tmp_path / "val.csv", tmp_path / "out")


# ---- weighted metrics -------------------------------------------------------------------------------------

def test_weighted_metrics_match_scikit_learn():
    rng = np.random.default_rng(3)
    y = rng.choice([-1, 0, 1], size=600)
    p = np.where(rng.random(600) < 0.3, rng.choice([-1, 0, 1], size=600), y)
    w = rng.uniform(0.2, 3.0, size=600)
    m = evaluate(y, p, sample_weight=w)
    assert m["weighted"] is True
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p, sample_weight=w))
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro", sample_weight=w))
    assert m["per_class"]["negative"]["support"] == pytest.approx(w[y == -1].sum())
    lo, hi = bootstrap_ci(y, p, sample_weight=w, n_boot=200)
    assert lo < m["macro_f1"] < hi


def test_bad_weights_rejected():
    with pytest.raises(ValueError, match="sample_weight"):
        evaluate([0, 1], [0, 1], sample_weight=[1.0])
    with pytest.raises(ValueError, match="sample_weight"):
        evaluate([0, 1], [0, 1], sample_weight=[1.0, -1.0])
