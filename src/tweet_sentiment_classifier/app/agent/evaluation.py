"""Evaluate the agent on a fixed fixture database with questions whose answers are computed from the data.

    python -m tweet_sentiment_classifier.app.agent.evaluation --out artifacts/agent_eval      # real model (needs a key)
    python -m tweet_sentiment_classifier.app.agent.evaluation --oracle                         # plumbing check only

Per question it scores: answer correctness (the numbers or names computed from the database appear in the answer),
tool selection (the expected tools were used), citation validity (nothing the agent cited had to be removed),
steps, tokens and latency. The oracle model replays the expected plan: it proves the harness, the tools and the
verifier work together, and says NOTHING about how good a real model is. Only report numbers from a run you made.

shortcut: 8 questions, not the ~25 the plan called for; add more to QUESTIONS (each is one entry).
"""
import argparse
import json
import random
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

from tweet_sentiment_classifier.app.agent import runner
from tweet_sentiment_classifier.app.core.llm.providers import LLMResponse
from tweet_sentiment_classifier.app.services import topic_analytics
from tweet_sentiment_classifier.entity.models import AgentRun, Alert, Insight, Post, Topic

TOPIC = "Fuel prices"
TEXTS = {
    "negative": ["fuel price hike hurts commuters", "petrol cost anger across the city", "diesel price surge angers drivers"],
    "neutral": ["fuel price announced for tuesday", "petrol pump timings revised this week", "diesel supply report released"],
    "positive": ["fuel price cut brings relief", "petrol discount welcomed by drivers", "good news on diesel supply"],
}


def build_fixture(db: Session, seed: int = 7) -> int:
    """14 days x 10 posts. The last 7 days are 40% negative, the 7 days before 20% negative."""
    rng = random.Random(seed)
    topic = Topic(name=TOPIC, query="fuel petrol diesel prices", sources=["google_news"])
    db.add(topic)
    db.commit()
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    n = 0
    for day in range(14):
        negatives = 4 if day < 7 else 2
        labels = ["negative"] * negatives + ["neutral"] * 3 + ["positive"] * (7 - negatives)
        rng.shuffle(labels)
        for label in labels:
            n += 1
            when = now - timedelta(days=day, hours=rng.randint(1, 10))
            db.add(Post(topic_id=topic.id, source="google_news", external_id=f"e{n}", content_hash=f"h{n}",
                        text=f"{rng.choice(TEXTS[label])} #{n}", prediction=label, confidence=0.9, published_at=when,
                        fetched_at=when))
    db.add(Alert(topic_id=topic.id, kind="negative_spike", window_hours=24, baseline=0.2, observed=0.4, p_value=0.002,
                 message="Negative share rose from 20% to 40%.", created_at=now))
    db.add(Insight(topic_id=topic.id, kind="themes", day=now.strftime("%Y-%m-%d"), created_at=now, payload={
        "n_posts": 140, "themes": [
            {"label": "Price hikes", "summary": "Complaints about rising prices.", "sentiment": "negative",
             "share": 0.6, "n": 84, "class_counts": {"negative": 84}, "post_ids": [1, 2, 3]},
            {"label": "Supply news", "summary": "Supply reports.", "sentiment": "neutral", "share": 0.4, "n": 56,
             "class_counts": {"neutral": 56}, "post_ids": [4, 5, 6]}]}))
    db.commit()
    return topic.id


def numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def has_number(text: str, value: float, tol: float) -> bool:
    return any(abs(x - value) <= tol for x in numbers(text))


@dataclass
class Question:
    question: str
    expected_tools: set
    plan: Callable[[int], List[tuple]]  # (tool, args) steps the oracle replays
    check: Callable[[Session, int, str, dict], bool]  # (db, topic_id, answer text, answer dict) -> correct?
    oracle_answer: Callable[[Session, int, str], str]  # what the oracle writes (it may read the last observation)


def _compare(db, tid, days=7):
    now = datetime.utcnow()
    when = func.coalesce(Post.published_at, Post.fetched_at)

    def share(start, end):
        q = db.query(Post.prediction, func.count(Post.id)).filter(Post.topic_id == tid, when >= start)
        if end:
            q = q.filter(when < end)
        c = dict(q.group_by(Post.prediction).all())
        return 100 * c.get("negative", 0) / max(1, sum(c.values()))
    return share(now - timedelta(days=days), None), share(now - timedelta(days=2 * days), now - timedelta(days=days))


def _cited_text_has(db, answer: dict, word: str) -> bool:
    return any(word in (db.get(Post, i).text if db.get(Post, i) else "") for i in answer.get("citations", []))


def _first_post_id(observation: str) -> int:
    return int(re.search(r'"id":(\d+)', observation).group(1))


QUESTIONS: List[Question] = [
    Question("How many posts are stored for the topic?", {"list_topics"},
             lambda t: [("list_topics", {})],
             lambda db, t, a, d: has_number(a, db.query(Post).filter(Post.topic_id == t).count(), 0),
             lambda db, t, obs: f"There are {db.query(Post).filter(Post.topic_id == t).count()} posts."),
    Question("What share of posts in the last 7 days were negative?", {"compare_periods"},
             lambda t: [("compare_periods", {"topic_id": t, "days": 7})],
             lambda db, t, a, d: has_number(a, _compare(db, t)[0], 2) or has_number(a, _compare(db, t)[0] / 100, 0.02),
             lambda db, t, obs: f"About {_compare(db, t)[0]:.0f}% were negative."),
    Question("What was the negative share in the 7 days before that?", {"compare_periods"},
             lambda t: [("compare_periods", {"topic_id": t, "days": 7})],
             lambda db, t, a, d: has_number(a, _compare(db, t)[1], 2) or has_number(a, _compare(db, t)[1] / 100, 0.02),
             lambda db, t, obs: f"About {_compare(db, t)[1]:.0f}% were negative."),
    Question("Which alert kinds were raised in the last week?", {"get_alerts"},
             lambda t: [("get_alerts", {"topic_id": t, "days": 7})],
             lambda db, t, a, d: "negative_spike" in a.replace(" ", "_").lower() or "negative spike" in a.lower(),
             lambda db, t, obs: "A negative_spike alert was raised."),
    Question("What is the largest theme and how many posts does it cover?", {"get_themes"},
             lambda t: [("get_themes", {"topic_id": t})],
             lambda db, t, a, d: "price hikes" in a.lower() and has_number(a, 84, 0),
             lambda db, t, obs: "The largest theme is Price hikes with 84 posts."),
    Question("How many posts were collected on the busiest day?", {"get_trend"},
             lambda t: [("get_trend", {"topic_id": t, "bucket": "day", "days": 14})],
             lambda db, t, a, d: has_number(a, max(p["n"] for p in topic_analytics.trends(db, t, "day", 14)), 0),
             lambda db, t, obs: f"{max(p['n'] for p in topic_analytics.trends(db, t, 'day', 14))} posts."),
    Question("Show me one post that complains about prices and cite it.", {"search_posts"},
             lambda t: [("search_posts", {"topic_id": t, "query": "price hike hurts anger", "sentiment": "negative"})],
             lambda db, t, a, d: bool(d.get("citations")) and _cited_text_has(db, d, "price"),
             lambda db, t, obs: f"For example [{_first_post_id(obs)}]."),
    Question("What sentiment does the model give the text 'good news on diesel supply'?", {"classify"},
             lambda t: [("classify", {"texts": ["good news on diesel supply"]})],
             lambda db, t, a, d: bool(re.search(r"positive|negative|neutral", a.lower())),
             lambda db, t, obs: "The model labels it " + re.search(r'"prediction":"(\w+)"', obs).group(1) + "."),
]


class Oracle:
    """Replays each question's expected plan, then writes the computed answer. Plumbing check only."""
    name, model = "oracle", "oracle-1"

    def __init__(self, Sess, topic_id, wrong: bool = False):
        self.Sess, self.topic_id, self.wrong = Sess, topic_id, wrong

    def complete(self, system, user, max_tokens=600):
        q = next(q for q in QUESTIONS if q.question in user)
        step = len(re.findall(r"^\[step \d+\]", user, flags=re.M))
        plan = q.plan(self.topic_id)
        if self.wrong:
            text = json.dumps({"action": "final", "answer": "I do not know.", "citations": []})
        elif step < len(plan):
            text = json.dumps({"action": "tool", "tool": plan[step][0], "args": plan[step][1]})
        else:
            with self.Sess() as db:
                obs = user[user.rfind("[step"):] if "[step" in user else ""
                answer = q.oracle_answer(db, self.topic_id, obs)
            cited = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
            text = json.dumps({"action": "final", "answer": answer, "citations": cited})
        return LLMResponse(text, 100, 40)


def run_eval(c, Sess: sessionmaker, topic_id: int, questions: List[Question] = QUESTIONS) -> Dict:
    rows = []
    for q in questions:
        with Sess() as db:
            run = AgentRun(kind="ask", question=q.question, topic_id=topic_id, status="running", steps=[])
            db.add(run)
            db.commit()
            runner.run_agent(db, c, run)
            answer = run.answer or {}
            used = {s["tool"] for s in run.steps if s["action"] == "tool"}
            rows.append({
                "question": q.question, "status": run.status,
                "correct": run.status == "done" and bool(q.check(db, topic_id, answer.get("text", ""), answer)),
                "tool_selection": q.expected_tools <= used, "tools": sorted(used),
                "citation_valid": run.status == "done" and not answer.get("removed_citations"),
                "steps": len(run.steps), "tokens": run.tokens_in + run.tokens_out, "latency_ms": run.latency_ms,
            })
    n = len(rows)
    mean = lambda key: round(statistics.fmean(r[key] for r in rows if r[key] is not None), 2)  # noqa: E731
    return {"n_questions": n,
            "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
            "tool_selection": round(sum(r["tool_selection"] for r in rows) / n, 3),
            "citation_validity": round(sum(r["citation_valid"] for r in rows) / n, 3),
            "mean_steps": mean("steps"), "mean_tokens": mean("tokens"), "mean_latency_ms": mean("latency_ms"),
            "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true", help="scripted plumbing check, not a model result")
    parser.add_argument("--out", help="directory for results.json (real-model runs)")
    args = parser.parse_args(argv)

    from tweet_sentiment_classifier.app.container import Container
    from tweet_sentiment_classifier.config.database import Base
    engine = create_engine(f"sqlite:///{Path(tempfile.mkdtemp()).as_posix()}/eval.db")
    Base.metadata.create_all(engine)
    Sess = sessionmaker(bind=engine)
    with Sess() as db:
        topic_id = build_fixture(db)

    c = Container.build()
    if args.oracle:
        c.llm.provider, c.llm.daily_token_budget = Oracle(Sess, topic_id), 10 ** 9
    elif not c.llm.configured:
        print("NOT RUN: no LLM is configured (set GROQ_API_KEY, or LLM_PROVIDER and its key).", file=sys.stderr)
        return 2
    report = run_eval(c, Sess, topic_id)
    report.update(model=f"{c.llm.provider.name}/{c.llm.provider.model}", date=date.today().isoformat(),
                  kind="oracle plumbing check (not a model result)" if args.oracle else "real model")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if args.out and not args.oracle:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
