"""Experiment scripts, run on toy data with fakes (no network, keys, GPU or real labels)."""
import json

import numpy as np
import pandas as pd
import pytest

from tweet_sentiment_classifier.experiments import behavioural, gold_budget, llm_eval, ood, report
from tweet_sentiment_classifier.experiments.cleanlab_run import run_cleanlab

VOCAB = {-1: "awful corrupt failure angry hate".split(), 0: "report meeting today city date".split(),
         1: "great happy success proud love".split()}


def toy(path, n=300, seed=0, flip=0.0):
    rng = np.random.default_rng(seed)
    ys = np.array([(-1, 0, 1)[i % 3] for i in range(n)])
    texts = [" ".join(rng.choice(VOCAB[y], 3)) + f" x{seed}_{i}" for i, y in enumerate(ys)]
    labels = np.where(rng.random(n) < flip, rng.choice([-1, 0, 1], size=n), ys)
    pd.DataFrame({"text": texts, "sentiment": labels.astype(float), "true": ys}).to_csv(path, index=False)
    return path


# ---- LLM evaluation --------------------------------------------------------------------------------

def test_llm_classifier_caches_retries_and_never_invents_labels(tmp_path):
    calls = {"n": 0}

    def complete(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 rate limited")
        return '{"label": 1, "explanation": "x"}' if "great" in prompt else "no idea"

    clf = llm_eval.CachedLLMClassifier(complete, "fake", tmp_path / "cache.jsonl", sleep=lambda s: None)
    assert clf.classify("great day") == 1                      # survived one rate-limit error
    assert clf.classify("hmm") is None                         # unparseable -> None, not neutral
    again = llm_eval.CachedLLMClassifier(complete, "fake", tmp_path / "cache.jsonl", sleep=lambda s: None)
    assert again.classify("great day") == 1 and again.api_calls == 0   # served from the cache


def test_evaluate_llm_reports_coverage_and_strict_accuracy(tmp_path):
    data = tmp_path / "d.csv"
    pd.DataFrame({"text": ["great a", "great b", "awful c", "meh d"], "sentiment": [1, 1, -1, 0]}).to_csv(data, index=False)
    replies = {"great": '{"label": 1}', "awful": '{"label": -1}'}
    complete = lambda p: next((v for k, v in replies.items() if k in p), "???")  # noqa: E731
    clf = llm_eval.CachedLLMClassifier(complete, "fake", tmp_path / "c.jsonl")
    r = llm_eval.evaluate_llm(data, tmp_path / "out", clf)
    assert (r["n"], r["n_valid"], r["coverage"]) == (4, 3, 0.75)
    assert r["strict_accuracy"] == 0.75 and r["valid_only"]["accuracy"] == 1.0


# ---- confident learning --------------------------------------------------------------------------------

def test_cleanlab_variants_with_a_fake_issue_finder(tmp_path):
    train = toy(tmp_path / "train.csv", n=300, flip=0.15)
    val = toy(tmp_path / "val.csv", n=90, seed=1)

    def finder(labels, probs):                                 # flag rows whose label disagrees with the model
        return probs.argmax(1) != labels

    r = run_cleanlab(train, {"val": val}, tmp_path / "out", folds=3, find_issues=finder)
    assert 0 < r["n_flagged"] < 300
    assert r["variants"]["drop"]["n_used"] == 300 - r["n_flagged"]
    assert r["variants"]["relabel"]["n_changed"] == r["n_flagged"]
    assert set(r["variants"]["baseline"]["scores"]) == {"val"}
    assert len(pd.read_csv(tmp_path / "out" / "flagged.csv")) == r["n_flagged"]


# ---- gold budget ---------------------------------------------------------------------------------------

def test_gold_budget_runs_every_k_and_seed(tmp_path):
    silver = toy(tmp_path / "silver.csv", n=240, flip=0.3)
    pool = toy(tmp_path / "pool.csv", n=120, seed=2)
    gtest = toy(tmp_path / "gtest.csv", n=90, seed=3)
    r = gold_budget.run_budget(silver, pool, gtest, tmp_path / "out", budgets=(0, 30, 60), seeds=(0, 1),
                               weights=(1.0, 10.0), params={"C": 4.0, "max_iter": 300}, n_boot=50,
                               progress=lambda *_: None)
    runs = pd.read_csv(tmp_path / "out" / "runs.csv")
    assert len(runs) == 6 and set(runs["k"]) == {0, 30, 60}
    assert runs.loc[runs["k"] == 0, "gold_only"].isna().all()
    assert runs.loc[runs["k"] > 0, "chosen_weight"].isin([1.0, 10.0]).all()
    assert len(r["summary"]) == 3
    with pytest.raises(ValueError, match="exceeds the gold pool"):
        gold_budget.run_budget(silver, pool, gtest, tmp_path / "o2", budgets=(500,), seeds=(0,), progress=lambda *_: None)


# ---- behavioural tests ----------------------------------------------------------------------------------

def test_generated_cases_need_review_before_scoring():
    cases = behavioural.generate_cases()
    assert cases["text"].is_unique and set(cases["type"]) == {"MFT", "INV"}
    with pytest.raises(ValueError, match="no reviewed cases"):
        behavioural.score_cases(cases, lambda texts: [0] * len(texts))


def test_scoring_mft_and_invariance():
    cases = pd.DataFrame([
        {"capability": "negation", "type": "MFT", "text": "not good", "expected": -1, "group": "", "reviewed": 1},
        {"capability": "negation", "type": "MFT", "text": "not bad", "expected": 1, "group": "", "reviewed": 1},
        {"capability": "entity_swap", "type": "INV", "text": "modi spoke", "expected": None, "group": "g1", "reviewed": 1},
        {"capability": "entity_swap", "type": "INV", "text": "rahul spoke", "expected": None, "group": "g1", "reviewed": 1},
    ])
    preds = {"not good": -1, "not bad": -1, "modi spoke": 1, "rahul spoke": 0}
    table = behavioural.score_cases(cases, lambda texts: [preds[t] for t in texts]).set_index("capability")
    assert table.loc["negation", "failures"] == 1 and table.loc["negation", "failure_rate"] == 0.5
    assert table.loc["entity_swap", "failures"] == 1                      # prediction changed with the name


# ---- out-of-domain ---------------------------------------------------------------------------------------

def test_ood_mapping_uses_the_dataset_label_names():
    # deliberately unusual order: id 0 = positive, 1 = negative, 2 = neutral
    df = ood.to_project_format(["Bad!", "OK.", "Great :)"], [1, 2, 0], ["positive", "negative", "neutral"])
    assert df["sentiment"].tolist() == [-1, 0, 1] and df["text"].tolist() == ["bad", "ok", "great"]
    with pytest.raises(ValueError, match="refusing to guess"):
        ood.to_project_format(["x"], [0], ["LABEL_0", "LABEL_1", "LABEL_2"])


# ---- report ----------------------------------------------------------------------------------------------

def fake_results(scores):
    return {"results": {m: {"macro_f1": s, "macro_f1_ci": (s - 0.01, s + 0.01)} for m, s in scores.items()}}


def test_table1_joins_silver_and_gold_and_computes_the_gap():
    table = report.silver_vs_gold_table(fake_results({"LinearSVC": 0.93, "TextBlob": 0.999}),
                                        fake_results({"LinearSVC": 0.70}))
    row = table.set_index("model").loc["LinearSVC"]
    assert row["gap"] == pytest.approx(0.23)
    assert pd.isna(table.set_index("model").loc["TextBlob"].get("gold_macro_f1"))
    tex = report.to_latex(table)
    assert "LinearSVC & 0.930 [0.920, 0.940] & 0.700 [0.690, 0.710] & 0.230" in tex


def test_error_sample_only_contains_errors(tmp_path):
    p = tmp_path / "pred.csv"
    pd.DataFrame({"text": list("abcd"), "label": [1, 0, -1, 1], "pred_LinearSVC": [1, 1, 0, 1]}).to_csv(p, index=False)
    sample = report.error_sample(p, "LinearSVC", n=10)
    assert sorted(sample["text"]) == ["b", "c"] and {"category", "notes"} <= set(sample.columns)


def test_budget_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    summary = [{"k": k, "silver_only_mean": 0.6, "silver_only_std": 0.0, "silver_plus_gold_mean": 0.6 + k / 1000,
                "silver_plus_gold_std": 0.01, "gold_only_mean": None, "gold_only_std": None} for k in (0, 50, 100)]
    report.budget_figure({"summary": summary}, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 1000


# ---- fine-tuning (needs torch) -------------------------------------------------------------------------------------

def test_finetune_saves_a_servable_three_label_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification
    from tweet_sentiment_classifier.models.finetune import TransformerModel, finetune

    base = tmp_path / "base"
    base.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted({w for ws in VOCAB.values() for w in ws})
    (base / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizer(str(base / "vocab.txt")).save_pretrained(base)
    torch.manual_seed(0)
    DistilBertForSequenceClassification(DistilBertConfig(vocab_size=len(vocab), dim=16, hidden_dim=32, n_layers=1,
                                                         n_heads=2, num_labels=2)).save_pretrained(base)
    log = finetune(str(base), toy(tmp_path / "tr.csv", 60), toy(tmp_path / "va.csv", 30, 1), tmp_path / "ft",
                   epochs=1, batch_size=16, device="cpu", progress=lambda *_: None)
    assert json.loads((tmp_path / "ft" / "config.json").read_text())["id2label"]["2"] == "positive"
    assert log["best_epoch"] == 1 and "test split not read" in log["note"]
    model = TransformerModel(str(tmp_path / "ft"), device="cpu")
    assert model.predict_proba(["great happy", "awful hate"]).shape == (2, 3)
