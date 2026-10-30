"""News sources over RSS. Standard library only (urllib + ElementTree), so they run everywhere, tests included.

To add a source (Reddit, YouTube, ...): write a class with `name` and `fetch(query) -> list[RawPost]`
and register it in `build_sources`.
"""
import hashlib
import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

from tweet_sentiment_classifier.utils.llm_safety import is_relevant

MAX_FEED_BYTES = 2_000_000
USER_AGENT = "tweet-sentiment-monitor/1.0"


@dataclass
class RawPost:
    source: str
    external_id: str
    url: str
    text: str
    published_at: datetime | None


def _plain(markup: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", markup or ""))).strip()


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _when(value: str | None) -> datetime | None:
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).replace(tzinfo=None) if value else None
    except (TypeError, ValueError):
        return None


def parse_rss(xml_bytes: bytes, source: str) -> list[RawPost]:
    # ElementTree expands entities, so a hostile feed could blow up memory: feeds have no use for them.
    if b"<!ENTITY" in xml_bytes or b"<!DOCTYPE" in xml_bytes:
        raise ValueError("feed rejected: it declares a DOCTYPE or entities")
    posts = []
    for item in ET.fromstring(xml_bytes).iter("item"):
        title, link = _plain(item.findtext("title")), (item.findtext("link") or "").strip()
        body = _plain(item.findtext("description"))
        same = not body or _norm(title) in _norm(body) or _norm(body) in _norm(title)  # Google repeats the title
        text = title if same else f"{title}. {body}"
        if not text:
            continue
        ident = (item.findtext("guid") or link or hashlib.sha1(text.encode()).hexdigest()).strip()
        posts.append(RawPost(source, ident, link, text, _when(item.findtext("pubDate"))))
    return posts


def fetch_url(url: str, timeout: float = 10.0) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("only https feeds are fetched")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as r:
        data = r.read(MAX_FEED_BYTES + 1)
    if len(data) > MAX_FEED_BYTES:
        raise ValueError("feed is larger than 2 MB")
    return data


class GoogleNews:
    name = "google_news"

    def fetch(self, query: str) -> list[RawPost]:
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        return parse_rss(fetch_url(url), self.name)


class RssFeeds:
    """Configured general feeds; only entries that mention the topic's query are kept."""
    name = "rss"

    def __init__(self, urls: list[str]):
        self.urls = urls

    def fetch(self, query: str) -> list[RawPost]:
        posts, errors = [], []
        for url in self.urls:
            try:
                posts += [p for p in parse_rss(fetch_url(url), self.name) if is_relevant(p.text, query)]
            except Exception as e:  # one dead feed must not hide the others
                errors.append(f"{url}: {e}")
        if errors and not posts:
            raise RuntimeError("; ".join(errors)[:300])
        return posts


def build_sources(rss_feeds: str = "") -> dict:
    sources = {"google_news": GoogleNews()}
    urls = [u.strip() for u in rss_feeds.split(",") if u.strip()]
    if urls:
        sources["rss"] = RssFeeds(urls)
    return sources
