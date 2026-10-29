"""Results page reads artifacts only; landing page makes no unsupported claims; DB commits do not block."""
import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def run_json(models):
    return {"run_id": "r1", "split": {"n_test": 10, "label_source": "x", "test_texts_also_in_train_pct": 0.0},
            "results": {m: {"macro_f1": f, "macro_f1_ci": [f - 0.01, f + 0.01], "accuracy": f, "kappa": f}
                        for m, f in models.items()},
            "comparisons": []}


def test_results_api_without_gold_reports_it_as_pending(env, tmp_path, monkeypatch):
    client, _ = env
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "artifacts" / "evaluation"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(run_json({"TextBlob": 0.999, "LinearSVC": 0.93})))
    body = client.get("/api/results").json()
    assert body["gold"] is None and body["disagreements"] == [] and body["seed_study"] is None
    rows = {r["model"]: r for r in body["silver"]["models"]}
    assert rows["TextBlob"]["baseline"] is True and rows["LinearSVC"]["baseline"] is False
    assert rows["LinearSVC"]["ci_low"] == 0.92


def test_results_api_with_gold_and_disagreements(env, tmp_path, monkeypatch):
    client, _ = env
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "artifacts" / "evaluation"
    (d / "gold").mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(run_json({"LinearSVC": 0.93})))
    (d / "gold" / "results.json").write_text(json.dumps(run_json({"LinearSVC": 0.70})))
    pd.DataFrame({"text": ["a", "b"], "label": [1, -1], "pred_TextBlob": [1, 0]}).to_csv(d / "gold" / "predictions.csv", index=False)
    body = client.get("/api/results").json()
    assert body["gold"]["models"][0]["macro_f1"] == 0.70
    assert body["disagreements"] == [{"text": "b", "human": "negative", "textblob": "neutral"}]


def test_results_page_renders_and_has_no_hardcoded_numbers(env):
    client, _ = env
    assert client.get("/results").status_code == 200
    html = (REPO / "templates" / "pages" / "results.html").read_text(encoding="utf-8")
    assert not re.search(r"\b0\.\d{2,}", html)                      # no metric literals in the template


def test_landing_page_has_no_unsupported_claims():
    html = (REPO / "templates" / "pages" / "landing.html").read_text(encoding="utf-8").lower()
    for claim in ("sub-100ms", "enterprise", "real-time", "gemma llm fallback", "predictive sentiment analytics"):
        assert claim not in html, claim


def test_slow_database_commit_does_not_block_the_event_loop(env, app_module, monkeypatch):
    from sqlalchemy.orm import Session
    original = Session.commit

    def slow_commit(self):
        time.sleep(0.5)
        return original(self)
    monkeypatch.setattr(Session, "commit", slow_commit)

    async def scenario():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            slow = asyncio.create_task(ac.post("/predict", params={"text": "slow commit please"}))
            await asyncio.sleep(0.1)
            t0 = time.perf_counter()
            health = await ac.get("/health")
            latency = time.perf_counter() - t0
            return (await slow), health, latency

    slow, health, latency = asyncio.run(scenario())
    assert slow.status_code == 200 and health.status_code == 200
    assert latency < 0.3, f"/health waited {latency:.2f}s behind a database commit"
