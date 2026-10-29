"""Headless tests for the Streamlit demo (skipped when streamlit is not installed)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "streamlit_demo" / "streamlit_app.py")
MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "best_model" / "best_model.pkl"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model artifacts not present")


def _app():
    return AppTest.from_file(APP, default_timeout=60).run()


def _analyze(at):
    next(b for b in at.button if "Analyze" in b.label).click().run()


def _prediction_html(at):
    return next(m.value for m in at.markdown if "PREDICTION" in m.value)


def test_app_loads_and_shows_label_caveat():
    at = _app()
    assert not at.exception
    assert any("TextBlob" in w.value for w in at.warning)


def test_sample_button_fills_text_and_analyze_predicts():
    at = _app()
    at.button[0].click().run()  # Positive sample
    assert "amazing" in at.text_area[0].value
    _analyze(at)
    assert not at.exception
    assert "Positive" in _prediction_html(at)


def test_typed_negative_text_is_negative():
    at = _app()
    at.text_area[0].set_value("worst decision ever, shameful and terrible").run()
    _analyze(at)
    assert "Negative" in _prediction_html(at)
    terms = next(d.value for d in at.dataframe if "term" in d.value.columns)
    assert terms.iloc[0]["term"] == "worst" and terms.iloc[0]["effect"] == "toward negative"


def test_empty_text_warns_instead_of_predicting():
    at = _app()
    _analyze(at)
    assert any("enter text" in w.value.lower() for w in at.warning)
