"""The only things the agent can do. Each tool validates its arguments, caps its output, and wraps an existing
service: the agent never writes SQL, never picks a URL, and only fetches the sources a topic already has.

Every tool returns a JSON-able dict. Posts it returns carry their database `id`, which is what an answer
may cite (see runner.verify_answer).
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import func
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app.services import ingestion_service, insight_service, topic_analytics
from tweet_sentiment_classifier.entity.models import Alert, Post, Topic

SENTIMENT = Literal["positive", "neutral", "negative", "uncertain"]
_WHEN = func.coalesce(Post.published_at, Post.fetched_at)
POOL = 1000  # shortcut: search ranks the newest 1000 posts of a topic; add embeddings/an index beyond that


class ToolError(Exception):
    """Bad arguments or a missing record: reported to the model as an observation, never raised to the user."""


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


class TopicArgs(Args):
    topic_id: int


class SearchArgs(Args):
    topic_id: int
    query: str = Field(min_length=1, max_length=200)
    sentiment: Optional[SENTIMENT] = None
    since_days: int = Field(default=7, ge=1, le=90)
    k: int = Field(default=5, ge=1, le=10)


class TrendArgs(Args):
    topic_id: int
    bucket: Literal["hour", "day"] = "day"
    days: int = Field(default=14, ge=1, le=30)


class PeriodArgs(Args):
    topic_id: int
    days: int = Field(default=7, ge=1, le=30)


class ClassifyArgs(Args):
    texts: List[str] = Field(min_length=1, max_length=10)


@dataclass
class Tool:
    name: str
    description: str
    args: Type[Args]
    fn: Callable[[Any], Dict[str, Any]]

    @property
    def signature(self) -> str:
        return f"{self.name}({', '.join(self.args.model_fields)}): {self.description}"


def _topic(db: Session, topic_id: int) -> Topic:
    topic = db.get(Topic, topic_id)
    if topic is None:
        raise ToolError(f"no topic with id {topic_id}")
    return topic


def post_json(p: Post) -> Dict[str, Any]:
    return {"id": p.id, "text": p.text[:280], "sentiment": p.prediction, "confidence": p.confidence,
            "source": p.source, "url": p.url,
            "published_at": p.published_at.isoformat() if p.published_at else None}


def build_tools(c, db: Session) -> Dict[str, Tool]:
    def list_topics(_: NoArgs) -> dict:
        counts = dict(db.query(Post.topic_id, func.count(Post.id)).group_by(Post.topic_id).all())
        return {"topics": [{"id": t.id, "name": t.name, "query": t.query, "posts": counts.get(t.id, 0)}
                           for t in db.query(Topic).order_by(Topic.id).all()]}

    def search_posts(a: SearchArgs) -> dict:
        _topic(db, a.topic_id)
        since = datetime.utcnow() - timedelta(days=a.since_days)
        query = db.query(Post).filter(Post.topic_id == a.topic_id, _WHEN >= since)
        if a.sentiment:
            query = query.filter(Post.prediction == a.sentiment)
        posts = query.order_by(Post.id.desc()).limit(POOL).all()
        if not posts:
            return {"posts": [], "note": "no posts match"}
        try:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            scores = (vec.fit_transform([p.text for p in posts]) @ vec.transform([a.query]).T).toarray().ravel()
        except ValueError:  # the query or the posts have no usable words
            return {"posts": [], "note": "no posts match"}
        order = [i for i in scores.argsort()[::-1][:a.k] if scores[i] > 0]
        return {"posts": [{**post_json(posts[i]), "score": round(float(scores[i]), 3)} for i in order]}

    def get_trend(a: TrendArgs) -> dict:
        _topic(db, a.topic_id)
        points = topic_analytics.trends(db, a.topic_id, a.bucket, a.days)
        return {"bucket": a.bucket, "points": [
            {"bucket": p["bucket"], "n": p["n"], "shares": {k: v["share"] for k, v in p["shares"].items()}}
            for p in points]}

    def compare_periods(a: PeriodArgs) -> dict:
        _topic(db, a.topic_id)
        now = datetime.utcnow()
        edges = [now - timedelta(days=2 * a.days), now - timedelta(days=a.days)]

        def counts(start, end):
            q = db.query(Post.prediction, func.count(Post.id)).filter(Post.topic_id == a.topic_id, _WHEN >= start)
            if end is not None:
                q = q.filter(_WHEN < end)
            return dict(q.group_by(Post.prediction).all())

        cur, prev = counts(edges[1], None), counts(edges[0], edges[1])
        n_cur, n_prev = sum(cur.values()), sum(prev.values())
        out: Dict[str, Any] = {"days": a.days, "current": {"n": n_cur, "counts": cur},
                               "previous": {"n": n_prev, "counts": prev}}
        if n_cur and n_prev:
            k1, k2 = cur.get("negative", 0), prev.get("negative", 0)
            out["negative_share"] = {"current": round(k1 / n_cur, 4), "previous": round(k2 / n_prev, 4),
                                     "p_value": round(topic_analytics.two_proportion_p(k1, n_cur, k2, n_prev), 4)}
        else:
            out["note"] = "one of the periods has no posts, so there is nothing to compare"
        return out

    def get_themes(a: TopicArgs) -> dict:
        _topic(db, a.topic_id)
        data = insight_service.latest_json(db, a.topic_id, "themes")
        if data is None:
            return {"themes": [], "note": "no themes have been built for this topic yet"}
        themes = []
        for t in data["themes"]:
            examples = db.query(Post).filter(Post.id.in_(t["post_ids"][:3])).all()
            themes.append({k: t[k] for k in ("label", "summary", "sentiment", "share", "n", "class_counts")}
                          | {"examples": [post_json(p) for p in examples]})
        return {"built_at": data["created_at"], "n_posts": data["n_posts"], "themes": themes}

    def get_alerts(a: PeriodArgs) -> dict:
        _topic(db, a.topic_id)
        rows = db.query(Alert).filter(Alert.topic_id == a.topic_id,
                                      Alert.created_at >= datetime.utcnow() - timedelta(days=a.days)
                                      ).order_by(Alert.id.desc()).limit(20).all()
        return {"alerts": [{"kind": r.kind, "message": r.message, "baseline": r.baseline, "observed": r.observed,
                            "p_value": r.p_value, "created_at": r.created_at.isoformat()} for r in rows]}

    def fetch_fresh(a: TopicArgs) -> dict:
        topic = _topic(db, a.topic_id)
        return {"runs": asyncio.run(ingestion_service.ingest_topic(c, db, topic)),
                "note": "only the topic's own configured sources were fetched"}

    def classify(a: ClassifyArgs) -> dict:
        if c.pipeline is None:
            raise ToolError("the prediction pipeline is not available")
        texts = [t.strip()[:c.settings.MAX_TEXT_CHARS] for t in a.texts if t.strip()]

        async def run():
            return [await c.pipeline.predict(t) for t in texts]
        return {"results": [{"text": t[:120], "prediction": r["prediction"], "confidence": r.get("confidence")}
                            for t, r in zip(texts, asyncio.run(run()))]}

    tools = [
        Tool("list_topics", "all monitored topics with their ids and post counts", NoArgs, list_topics),
        Tool("search_posts", "keyword search over a topic's recent posts; returns posts with ids you can cite",
             SearchArgs, search_posts),
        Tool("get_trend", "posts per day or hour with the share of each sentiment", TrendArgs, get_trend),
        Tool("compare_periods", "the last N days against the N days before: counts and the negative share with a "
             "p-value", PeriodArgs, compare_periods),
        Tool("get_themes", "the latest grouped themes of a topic with example posts you can cite", TopicArgs,
             get_themes),
        Tool("get_alerts", "alerts raised for a topic in the last N days", PeriodArgs, get_alerts),
        Tool("fetch_fresh", "collect new posts for a topic now (once per session)", TopicArgs, fetch_fresh),
        Tool("classify", "sentiment of up to 10 texts you supply", ClassifyArgs, classify),
    ]
    return {t.name: t for t in tools}
