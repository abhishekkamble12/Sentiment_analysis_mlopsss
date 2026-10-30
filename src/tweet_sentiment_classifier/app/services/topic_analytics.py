"""Trends, drift and alerts for a topic, computed from its stored posts.

Time of a post = when it was published, or when we fetched it if the feed gave no date.
A "window" is the last ALERT_WINDOW_HOURS; its baseline is the 7 days before the window.

What the drift numbers can and cannot say:
- OOV rate is the share of words the model never saw (outside its 15,000-word vocabulary). A high absolute
  value means "different vocabulary from the training tweets"; a rise against the baseline means the
  topic is changing. There is no stored training-set reference, so no absolute "normal" is claimed.
- PSI compares the class mix of the window with its baseline (0.1 small, 0.25 large shift).
- Agreement with an LLM on a sample (the live quality signal in the plan) needs the Phase 5 provider layer.
"""
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.components.data_transformation import clean_text
from tweet_sentiment_classifier.entity.models import Alert, Post, Topic

CLASSES = ("positive", "neutral", "negative", "uncertain")
BASELINE_DAYS = 7
MAX_ROWS = 20000  # shortcut: aggregation runs in Python over the newest rows; move it to SQL if a topic exceeds this
_WHEN = func.coalesce(Post.published_at, Post.fetched_at)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value of the z-test that k1/n1 and k2/n2 are the same proportion."""
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return math.erfc(abs(k1 / n1 - k2 / n2) / se / math.sqrt(2))


def psi(current: Dict[str, int], baseline: Dict[str, int], eps: float = 1e-4) -> float:
    """Population stability index between two class counts (eps keeps empty classes finite)."""
    cn, bn = sum(current.values()) or 1, sum(baseline.values()) or 1
    total = 0.0
    for cls in CLASSES:
        a, b = max(current.get(cls, 0) / cn, eps), max(baseline.get(cls, 0) / bn, eps)
        total += (a - b) * math.log(a / b)
    return total


def oov_rate(vectorizer, texts: List[str]) -> Optional[float]:
    vocab = getattr(vectorizer, "vocabulary_", None)
    if not vocab:
        return None
    tokens = [t for text in texts for t in re.findall(r"\b\w\w+\b", clean_text(text))]
    return round(sum(t not in vocab for t in tokens) / len(tokens), 4) if tokens else None


def trends(db: Session, topic_id: int, bucket: str = "day", days: int = 14,
           now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    fmt = "%Y-%m-%d %H:00" if bucket == "hour" else "%Y-%m-%d"
    since = (now or datetime.utcnow()) - timedelta(days=days)
    rows = (db.query(_WHEN, Post.prediction, Post.confidence, Post.source)
            .filter(Post.topic_id == topic_id, _WHEN >= since).order_by(_WHEN.desc()).limit(MAX_ROWS).all())
    groups: Dict[str, list] = defaultdict(list)
    for when, prediction, confidence, source in rows:
        groups[when.strftime(fmt)].append((prediction, confidence, source))

    out = []
    for key in sorted(groups):
        items = groups[key]
        n, counts = len(items), Counter(p for p, _, _ in items)
        confs = [c for _, c, _ in items if c is not None]
        out.append({
            "bucket": key, "n": n, "counts": {c: counts.get(c, 0) for c in CLASSES},
            "shares": {c: {"share": round(counts.get(c, 0) / n, 4),
                           "low": round(wilson(counts.get(c, 0), n)[0], 4),
                           "high": round(wilson(counts.get(c, 0), n)[1], 4)} for c in CLASSES},
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "by_source": dict(Counter(s for _, _, s in items)),
        })
    return out


def _window(db: Session, topic_id: int, start: datetime, end: Optional[datetime], vectorizer) -> Dict[str, Any]:
    query = db.query(_WHEN, Post.prediction, Post.confidence, Post.text).filter(
        Post.topic_id == topic_id, _WHEN >= start)
    if end is not None:
        query = query.filter(_WHEN < end)
    rows = query.limit(MAX_ROWS).all()
    n, counts = len(rows), Counter(r[1] for r in rows)
    confs = [r[2] for r in rows if r[2] is not None]
    return {
        "start": start.isoformat() + "Z", "n": n, "counts": {c: counts.get(c, 0) for c in CLASSES},
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "uncertain_share": round(counts.get("uncertain", 0) / n, 4) if n else None,
        "oov_rate": oov_rate(vectorizer, [r[3] for r in rows]) if n else None,
        "first": min((r[0] for r in rows), default=None),
    }


def drift_report(db: Session, topic_id: int, vectorizer, window_hours: int = 24,
                 now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.utcnow()
    cut = now - timedelta(hours=window_hours)
    current = _window(db, topic_id, cut, None, vectorizer)
    baseline = _window(db, topic_id, cut - timedelta(days=BASELINE_DAYS), cut, vectorizer)
    both = current["n"] and baseline["n"]
    first = baseline.pop("first")
    current.pop("first")
    baseline["days_covered"] = round(min(BASELINE_DAYS, max(1.0, (cut - first).total_seconds() / 86400)), 2) if first else 0
    return {"window_hours": window_hours, "window": current, "baseline": baseline,
            "psi": round(psi(current["counts"], baseline["counts"]), 4) if both else None}


def evaluate_alerts(db: Session, topic: Topic, vectorizer, settings, now: Optional[datetime] = None) -> List[Alert]:
    """Create alerts for unusual changes. Needs ALERT_MIN_POSTS in both windows; one alert per kind per cooldown."""
    now = now or datetime.utcnow()
    report = drift_report(db, topic.id, vectorizer, settings.ALERT_WINDOW_HOURS, now)
    cur, base = report["window"], report["baseline"]
    if min(cur["n"], base["n"]) < settings.ALERT_MIN_POSTS:
        return []

    found: List[Alert] = []

    def add(kind: str, baseline: float, observed: float, p: Optional[float], message: str) -> None:
        cooling = db.query(Alert.id).filter(
            Alert.topic_id == topic.id, Alert.kind == kind,
            Alert.created_at > now - timedelta(hours=settings.ALERT_COOLDOWN_HOURS)).first()
        if cooling:
            return
        alert = Alert(topic_id=topic.id, kind=kind, window_hours=settings.ALERT_WINDOW_HOURS, baseline=baseline,
                      observed=observed, p_value=p, message=message, created_at=now)
        db.add(alert)
        found.append(alert)

    neg_cur, neg_base = cur["counts"]["negative"], base["counts"]["negative"]
    share_cur, share_base = neg_cur / cur["n"], neg_base / base["n"]
    p = two_proportion_p(neg_cur, cur["n"], neg_base, base["n"])
    if share_cur > share_base and p < settings.ALERT_P_VALUE:
        add("negative_spike", share_base, share_cur, p,
            f"Negative share rose from {share_base:.0%} to {share_cur:.0%} "
            f"({cur['n']} posts in the last {settings.ALERT_WINDOW_HOURS} h, p={p:.4f}).")

    per_window = base["n"] / base["days_covered"] * settings.ALERT_WINDOW_HOURS / 24
    if cur["n"] > 3 * max(per_window, 1):
        add("volume_spike", per_window, cur["n"], None,
            f"{cur['n']} posts in the last {settings.ALERT_WINDOW_HOURS} h against about {per_window:.0f} usually.")

    if report["psi"] is not None and report["psi"] > settings.DRIFT_PSI_ALERT:
        add("class_drift", 0.0, report["psi"], None,
            f"The mix of positive, neutral and negative posts shifted (PSI {report['psi']:.2f}).")

    db.commit()
    return found
