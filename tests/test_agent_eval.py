"""The agent evaluation harness: fixture truth, scoring that can fail, and the oracle plumbing check."""
import json

import pytest

from tests.test_agent import use_model
from tweet_sentiment_classifier.app.agent import evaluation as ev
from tweet_sentiment_classifier.entity.models import Post


@pytest.fixture
def fixture_db(env):
    _, Session = env
    with Session() as db:
        tid = ev.build_fixture(db)
    return Session, tid


def test_fixture_has_the_designed_shape(fixture_db):
    Session, tid = fixture_db
    with Session() as db:
        assert db.query(Post).filter(Post.topic_id == tid).count() == 140
        this_week, last_week = ev._compare(db, tid)
        assert (round(this_week), round(last_week)) == (40, 20)


def test_has_number_matches_within_tolerance():
    assert ev.has_number("About 40% were negative", 40, 0) and ev.has_number("39.6 percent", 40, 1)
    assert not ev.has_number("about forty", 40, 5) and not ev.has_number("41 posts", 40, 0)


def test_oracle_passes_every_question_which_proves_the_plumbing(fixture_db, app_module, monkeypatch):
    Session, tid = fixture_db
    use_model(app_module, monkeypatch, ev.Oracle(Session, tid))
    report = ev.run_eval(app_module, Session, tid)
    failed = [r for r in report["rows"] if not (r["correct"] and r["tool_selection"] and r["citation_valid"])]
    assert not failed, failed
    assert report["n_questions"] == len(ev.QUESTIONS) and report["accuracy"] == 1.0
    assert report["mean_steps"] >= 2 and report["mean_tokens"] > 0


def test_a_model_that_never_looks_scores_zero(fixture_db, app_module, monkeypatch):
    """The scoring must be able to fail: an agent that answers without using any tool is wrong and unverified."""
    Session, tid = fixture_db
    use_model(app_module, monkeypatch, ev.Oracle(Session, tid, wrong=True))
    report = ev.run_eval(app_module, Session, tid)
    assert report["accuracy"] == 0.0 and report["tool_selection"] == 0.0


def test_cli_oracle_run_end_to_end(capsys):
    assert ev.main(["--oracle"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["accuracy"] == 1.0 and out["kind"].startswith("oracle plumbing check")
