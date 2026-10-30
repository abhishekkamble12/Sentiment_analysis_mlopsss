"""Themes and the daily digest for a topic.

Posts of the last THEME_WINDOW_HOURS are grouped by their wording; the LLM (when configured) only names and
summarises each group. Sentiment is never taken from the LLM: it is the model's own labels for the group's
posts. Quotes must be exact substrings of a member post, otherwise they are dropped.

shortcut: grouping is TF-IDF + SVD + KMeans with k from a sqrt rule, not sentence embeddings + HDBSCAN. It
needs no model download; switch when Phase 6 adds embeddings for search.
"""
import logging
import math
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sqlalchemy import func
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app import prompts
from tweet_sentiment_classifier.app.core.llm.client import BudgetExceeded
from tweet_sentiment_classifier.entity.models import Alert, Insight, Post, Topic
from tweet_sentiment_classifier.utils.llm_safety import sanitize_untrusted, wrap_untrusted

logger = logging.getLogger("api")
CLASSES = ("positive", "neutral", "negative", "uncertain")
MIN_CLUSTER = 3
MAX_POSTS = 1000  # shortcut: cluster the newest posts only; raise it if clustering stays fast enough


class ThemeAnswer(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    summary: str = Field(min_length=1, max_length=400)
    quotes: List[str] = Field(default_factory=list, max_length=3)


class DigestAnswer(BaseModel):
    summary: str = Field(min_length=1, max_length=900)
    changes: List[str] = Field(default_factory=list, max_length=5)


def cluster_posts(texts: List[str], seed: int = 0) -> tuple[np.ndarray, List[List[str]], np.ndarray]:
    """(cluster id per text, top words per cluster, distance of each text to its cluster centre)."""
    n = len(texts)
    k = max(2, min(8, round(math.sqrt(n / 2))))
    try:
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=5000,
                                sublinear_tf=True)
        x = tfidf.fit_transform(texts)
    except ValueError:  # no word appears twice: nothing to group by
        return np.zeros(n, dtype=int), [[]], np.zeros(n)
    components = min(50, x.shape[1] - 1, n - 1)
    if components < 2:
        return np.zeros(n, dtype=int), [[]], np.zeros(n)
    z = Normalizer().fit_transform(TruncatedSVD(components, random_state=seed).fit_transform(x))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
    names = tfidf.get_feature_names_out()
    top = []
    for c in range(k):
        mean = np.asarray(x[km.labels_ == c].mean(axis=0)).ravel()
        top.append([str(names[j]) for j in mean.argsort()[::-1][:3] if mean[j] > 0])
    distance = np.linalg.norm(z - km.cluster_centers_[km.labels_], axis=1)
    return km.labels_, top, distance


def _verified_quotes(quotes: List[str], member_texts: List[str]) -> List[str]:
    """Only phrases that really occur in a member post (an LLM must not put words in people's mouths)."""
    kept = []
    for q in quotes:
        q = q.strip()
        if q and any(q in text for text in member_texts) and q not in kept:
            kept.append(q)
    return kept[:2]


def _name_theme(client, db: Session, topic: Topic, examples: List[str], now) -> Optional[ThemeAnswer]:
    user = (f"Topic: {sanitize_untrusted(topic.name, 100)}\nPosts in one group:\n"
            + wrap_untrusted("data", "\n".join(f"{i + 1}. {sanitize_untrusted(t, 280)}"
                                               for i, t in enumerate(examples)), 2400))
    return client.call_json(db, prompt_version=prompts.THEME_VERSION, system=prompts.THEME_SYSTEM, user=user,
                            model_cls=ThemeAnswer, max_tokens=300, now=now)


def build_themes(db: Session, topic: Topic, client, settings, now: Optional[datetime] = None) -> Optional[Insight]:
    now = now or datetime.utcnow()
    since = now - timedelta(hours=settings.THEME_WINDOW_HOURS)
    posts = (db.query(Post).filter(Post.topic_id == topic.id, func.coalesce(Post.published_at, Post.fetched_at) >= since)
             .order_by(Post.id.desc()).limit(MAX_POSTS).all())
    if len(posts) < settings.INSIGHT_MIN_POSTS:
        return None

    texts = [p.text for p in posts]
    labels, top_words, distance = cluster_posts(texts)
    themes, unclustered, budget_exhausted = [], 0, False
    for cid in sorted(set(labels.tolist()), key=lambda c: -int((labels == c).sum())):
        members = [i for i in np.where(labels == cid)[0]]
        if len(members) < MIN_CLUSTER:
            unclustered += len(members)
            continue
        members.sort(key=lambda i: distance[i])
        member_texts = [texts[i] for i in members]
        counts = Counter(posts[i].prediction for i in members)
        theme: Dict[str, Any] = {
            "label": " / ".join(top_words[cid]) or "other", "summary": "", "quotes": [], "llm": False,
            "n": len(members), "share": round(len(members) / len(posts), 4),
            "sentiment": max(CLASSES, key=lambda c: counts.get(c, 0)),
            "class_counts": {c: counts.get(c, 0) for c in CLASSES},
            "post_ids": [posts[i].id for i in members[:5]],
        }
        if client is not None and client.configured and not budget_exhausted:
            try:
                answer = _name_theme(client, db, topic, member_texts[:6], now)
            except BudgetExceeded as e:
                logger.warning(str(e))
                budget_exhausted = True
                answer = None
            if answer is not None:
                theme.update(label=answer.label, summary=answer.summary, llm=True,
                             quotes=_verified_quotes(answer.quotes, member_texts))
        themes.append(theme)

    payload = {"window_hours": settings.THEME_WINDOW_HOURS, "n_posts": len(posts), "unclustered": unclustered,
               "budget_exhausted": budget_exhausted, "themes": themes,
               "llm": {"provider": client.provider.name, "model": client.provider.model,
                       "prompt_version": prompts.THEME_VERSION} if client is not None and client.configured else None}
    row = Insight(topic_id=topic.id, kind="themes", day=now.strftime("%Y-%m-%d"), payload=payload, created_at=now)
    db.add(row)
    db.commit()
    return row


def _latest(db: Session, topic_id: int, kind: str, before_day: Optional[str] = None,
            day: Optional[str] = None) -> Optional[Insight]:
    query = db.query(Insight).filter(Insight.topic_id == topic_id, Insight.kind == kind)
    if before_day:
        query = query.filter(Insight.day < before_day)
    if day:
        query = query.filter(Insight.day == day)
    return query.order_by(Insight.id.desc()).first()


def _theme_lines(row: Optional[Insight]) -> str:
    if row is None:
        return "(no data)"
    total = row.payload["n_posts"]
    return "\n".join(f"- {sanitize_untrusted(t['label'], 80)}: {t['sentiment']}, {t['share']:.0%} of {total} posts"
                     for t in row.payload["themes"]) or "(no themes)"


def build_digest(db: Session, topic: Topic, client, now: Optional[datetime] = None) -> Optional[Insight]:
    """One LLM call comparing today's themes with the last earlier day's. None without an LLM or themes."""
    now = now or datetime.utcnow()
    if client is None or not client.configured:
        return None
    today = now.strftime("%Y-%m-%d")
    current = _latest(db, topic.id, "themes", day=today)
    if current is None or not current.payload["themes"]:
        return None
    previous = _latest(db, topic.id, "themes", before_day=today)
    alerts = db.query(Alert).filter(Alert.topic_id == topic.id, Alert.created_at >= now - timedelta(hours=24)).all()
    user = (f"Topic: {sanitize_untrusted(topic.name, 100)}\n"
            + wrap_untrusted("data", f"TODAY\n{_theme_lines(current)}\n\nYESTERDAY\n{_theme_lines(previous)}\n\n"
                                     "ALERTS IN THE LAST DAY\n"
                                     + ("\n".join(f"- {a.message}" for a in alerts) or "(none)"), 4000))
    try:
        answer = client.call_json(db, prompt_version=prompts.DIGEST_VERSION, system=prompts.DIGEST_SYSTEM,
                                  user=user, model_cls=DigestAnswer, max_tokens=500, now=now)
    except BudgetExceeded as e:
        logger.warning(str(e))
        return None
    if answer is None:
        return None
    payload = {**answer.model_dump(), "had_previous_day": previous is not None,
               "links": [{"theme": t["label"], "post_ids": t["post_ids"]} for t in current.payload["themes"]],
               "llm": {"provider": client.provider.name, "model": client.provider.model,
                       "prompt_version": prompts.DIGEST_VERSION}}
    row = Insight(topic_id=topic.id, kind="digest", day=today, payload=payload, created_at=now)
    db.add(row)
    db.commit()
    return row


def refresh_topic(db: Session, topic: Topic, client, settings, now: Optional[datetime] = None,
                  force: bool = False) -> Dict[str, Any]:
    """Themes at most every INSIGHT_INTERVAL_HOURS, and one digest per day (also what bounds LLM spend)."""
    now = now or datetime.utcnow()
    last = _latest(db, topic.id, "themes")
    if not force and last is not None and now - last.created_at < timedelta(hours=settings.INSIGHT_INTERVAL_HOURS):
        return {"themes": "skipped", "digest": "skipped"}
    themes = build_themes(db, topic, client, settings, now)
    today = now.strftime("%Y-%m-%d")
    digest = None
    if themes is not None and (force or _latest(db, topic.id, "digest", day=today) is None):
        digest = build_digest(db, topic, client, now)
    return {"themes": "built" if themes else "not enough posts", "digest": "built" if digest else "none"}


def to_json(row: Optional[Insight]) -> Optional[Dict[str, Any]]:
    return None if row is None else {**row.payload, "created_at": row.created_at.isoformat() + "Z", "day": row.day}


def latest_json(db: Session, topic_id: int, kind: str) -> Optional[Dict[str, Any]]:
    return to_json(_latest(db, topic_id, kind))

