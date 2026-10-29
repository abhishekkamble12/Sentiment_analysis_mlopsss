"""The response must always say who actually decided (audit finding B2)."""
import asyncio

import pytest

from conftest import StubMarginModel, StubProbaModel
from tweet_sentiment_classifier.pipeline.prediction_pipeline import PredictionPipeline

LLM_OK = {"label": -1, "explanation": "angry tone", "source": "llm:groq/llama-3.1-8b-instant"}


def run(pipeline, text="some tweet"):
    return asyncio.run(pipeline.predict(text))


def test_local_mode_never_calls_llm(make_pipeline):
    p = make_pipeline("local", StubProbaModel([0.1, 0.2, 0.7]), llm=LLM_OK)
    r = run(p)
    assert p.llm_calls == 0
    assert r["decision_source"] == r["model_used"] == "local:logisticregression"
    assert r["prediction"] == "positive"
    assert r["confidence"] == pytest.approx(0.7)
    assert r["explanation"] is None
    assert r["fallback_triggered"] is False


def test_llm_mode_reports_llm_and_no_fake_confidence(make_pipeline):
    p = make_pipeline("llm", StubProbaModel([0.1, 0.2, 0.7]), llm=LLM_OK)
    r = run(p)
    assert r["decision_source"] == "llm:groq/llama-3.1-8b-instant"
    assert r["prediction"] == "negative"  # the LLM's label wins, local vote is positive
    assert r["confidence"] is None  # the old code hardcoded 0.9
    assert r["explanation"] == "angry tone"
    assert r["votes"]["local:logisticregression"]["label"] == "positive"
    assert r["votes"]["llm:groq/llama-3.1-8b-instant"]["label"] == "negative"
    assert r["fallback_triggered"] is False  # the LLM is the primary in this mode


def test_llm_mode_failure_falls_back_to_local_and_says_so(make_pipeline):
    p = make_pipeline("llm", StubProbaModel([0.1, 0.2, 0.7]), llm=None)
    r = run(p)
    assert p.llm_calls == 1
    assert r["decision_source"] == "local:logisticregression"
    assert r["fallback_triggered"] is True
    assert "llm_unavailable" in r["warnings"]


def test_hybrid_confident_local_skips_llm(make_pipeline):
    p = make_pipeline("hybrid", StubProbaModel([0.05, 0.05, 0.9]), threshold=0.75, llm=LLM_OK)
    r = run(p)
    assert p.llm_calls == 0
    assert r["decision_source"] == "local:logisticregression"
    assert r["fallback_triggered"] is False


def test_hybrid_low_confidence_escalates_to_llm(make_pipeline):
    p = make_pipeline("hybrid", StubProbaModel([0.3, 0.3, 0.4]), threshold=0.75, llm=LLM_OK)
    r = run(p)
    assert p.llm_calls == 1
    assert r["decision_source"].startswith("llm:")
    assert r["fallback_triggered"] is True


def test_broken_fast_model_is_reported_not_masked_as_neutral(make_pipeline):
    # Old behaviour: the LinearSVC confidence crash became neutral/0.0. Now it is a visible failure.
    p = make_pipeline("local", StubMarginModel())
    with pytest.raises(RuntimeError, match="No model available"):
        run(p)


def test_failed_fast_model_surfaces_warning_when_distilbert_answers(make_pipeline):
    p = make_pipeline("local", StubMarginModel(), distilbert=(1, 0.8))
    r = run(p)
    assert r["decision_source"] == "local:distilbert"
    assert "local_fast_failed" in r["warnings"]


def test_no_model_and_no_llm_raises(make_pipeline):
    p = make_pipeline("hybrid", fast_model=None, llm=None)
    with pytest.raises(RuntimeError):
        run(p)


@pytest.mark.parametrize("text,expected", [
    ("Label: 1\nExplanation: great", (1, "great")),
    ("Label: -1\nExplanation: bad", (-1, "bad")),
    ("Label: 0", (0, "")),
    ("Label: 2", None),
    ("Label: 10", None),
    ("positive!", None),
    ("", None),
    (None, None),
])
def test_parse_llm_response(text, expected):
    assert PredictionPipeline._parse_llm_response(text) == expected


def test_invalid_inference_mode_rejected():
    with pytest.raises(ValueError, match="INFERENCE_MODE"):
        PredictionPipeline(inference_mode="ensemble")
