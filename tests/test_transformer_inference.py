"""Run the real PyTorch loader/predictor on a tiny random 3-label checkpoint.

Skipped where torch/transformers are not installed (CI installs neither), so it runs in the serving
environment (requirements/serve.txt) only. The weights are random: this checks wiring and shapes,
not model quality.
"""
import asyncio

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import StubProbaModel  # noqa: E402

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "great", "awful", "the", "is", "day"]


def build_checkpoint(path, num_labels):
    from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification
    path.mkdir(parents=True)
    (path / "vocab.txt").write_text("\n".join(VOCAB))
    BertTokenizer(str(path / "vocab.txt"), do_lower_case=True).save_pretrained(path)
    cfg = DistilBertConfig(vocab_size=len(VOCAB), dim=16, hidden_dim=32, n_layers=1, n_heads=2,
                           max_position_embeddings=64, num_labels=num_labels)
    torch.manual_seed(0)
    DistilBertForSequenceClassification(cfg).save_pretrained(path)


@pytest.fixture
def tier2_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path / "artifacts" / "model_trainer" / "model_DistilBERT"


def test_three_class_checkpoint_loads_and_predicts(make_pipeline, tier2_dir):
    build_checkpoint(tier2_dir, 3)
    p = make_pipeline("local")
    p._load_models()
    assert p.db_model is not None and not p.db_model.training  # loaded and in eval mode

    label, conf = p._predict_distilbert("the day is good")
    assert label in (-1, 0, 1) and 0.0 < conf <= 1.0
    assert p._predict_distilbert("the day is good") == (label, conf)  # deterministic in eval mode


def test_predict_includes_the_transformer_vote(make_pipeline, tier2_dir):
    build_checkpoint(tier2_dir, 3)
    p = make_pipeline("local", StubProbaModel([0.1, 0.1, 0.8]))
    p._load_models()
    p.fast_model = StubProbaModel([0.1, 0.1, 0.8])  # _load_models found no fast-model pickle
    r = asyncio.run(p.predict("the day is good"))
    assert r["decision_source"] == "local:logisticregression"
    assert set(r["votes"]) == {"local:logisticregression", "local:distilbert"}
    assert r["votes"]["local:distilbert"]["label"] in ("negative", "neutral", "positive")


def test_two_class_checkpoint_is_refused_for_real(make_pipeline, tier2_dir):
    build_checkpoint(tier2_dir, 2)
    p = make_pipeline("local")
    p._load_models()
    assert p.db_model is None and p.db_tokenizer is None
