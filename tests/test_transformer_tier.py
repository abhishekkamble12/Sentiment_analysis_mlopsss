"""The transformer tier must be 3-class or disabled; it must never fall back to a 2-class model (audit B4)."""
import json
import sys
import types

import pytest

from tweet_sentiment_classifier.pipeline.prediction_pipeline import transformer_num_labels

TIER2 = ("artifacts", "model_trainer", "model_DistilBERT")


@pytest.fixture
def fake_transformers(monkeypatch):
    calls = []

    class _Model:
        def eval(self):
            return self

    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: calls.append(("tok", path)) or object())
    mod.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda path: calls.append(("model", path)) or _Model()
    )
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return calls


def write_config(root, cfg):
    d = root.joinpath(*TIER2)
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # _load_models uses paths relative to the working directory
    return tmp_path


def test_missing_model_disables_tier_without_downloading(make_pipeline, in_tmp, fake_transformers):
    p = make_pipeline("local")
    p._load_models()
    assert p.db_model is None and p.db_tokenizer is None
    assert fake_transformers == []  # nothing was fetched or loaded


def test_two_class_checkpoint_is_refused(make_pipeline, in_tmp, fake_transformers):
    write_config(in_tmp, {"id2label": {"0": "NEGATIVE", "1": "POSITIVE"}})
    p = make_pipeline("local")
    p._load_models()
    assert p.db_model is None
    assert fake_transformers == []


def test_three_class_checkpoint_is_loaded(make_pipeline, in_tmp, fake_transformers):
    write_config(in_tmp, {"id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}})
    p = make_pipeline("local")
    p._load_models()
    assert p.db_model is not None and p.db_tokenizer is not None
    assert [c[0] for c in fake_transformers] == ["tok", "model"]


def test_unloadable_checkpoint_disables_tier(make_pipeline, in_tmp, monkeypatch):
    write_config(in_tmp, {"num_labels": 3})
    mod = types.ModuleType("transformers")

    def boom(path):
        raise OSError("corrupt weights")
    mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=boom)
    mod.AutoModelForSequenceClassification = types.SimpleNamespace(from_pretrained=boom)
    monkeypatch.setitem(sys.modules, "transformers", mod)
    p = make_pipeline("local")
    p._load_models()
    assert p.db_model is None and p.db_tokenizer is None


def test_disabled_tier_is_reported_as_failure_not_a_neutral_vote(make_pipeline, in_tmp):
    p = make_pipeline("local")
    with pytest.raises(RuntimeError, match="not loaded"):
        p._predict_distilbert("hello")


def test_disabled_tier_is_skipped_without_a_failure_warning(make_pipeline):
    import asyncio
    from conftest import StubProbaModel
    res = asyncio.run(make_pipeline("local", fast_model=StubProbaModel([0.1, 0.1, 0.8])).predict("great"))
    assert res["warnings"] == [] and list(res["votes"]) == ["local:logisticregression"]


@pytest.mark.parametrize("cfg,expected", [
    ({"id2label": {"0": "a", "1": "b", "2": "c"}}, 3),
    ({"id2label": {"0": "a", "1": "b"}}, 2),
    ({"num_labels": 3}, 3),
    ({}, None),
])
def test_transformer_num_labels(tmp_path, cfg, expected):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert transformer_num_labels(tmp_path) == expected


def test_transformer_num_labels_unreadable(tmp_path):
    assert transformer_num_labels(tmp_path) is None  # no config.json
    (tmp_path / "config.json").write_text("{not json")
    assert transformer_num_labels(tmp_path) is None
