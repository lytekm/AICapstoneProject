from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import re
import urllib.request

import feedparser

# Optional but recommended: robust article extraction
# pip install trafilatura
try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None


@dataclass
class Article:
    title: str
    url: str
    published: Optional[str]
    fetched_at: str
    raw_text: str
    normalized_text: str
    sentence_count: int
    word_count: int


class NewsDataPipeline:
    """
    Data Engineering + Pipeline module (tokenize + normalize included).

    Responsibilities:
    - Fetch RSS feed entries (title, link, published)
    - Fetch article HTML and extract text
    - Normalize text
    - Tokenize (sentence count / word count) for stats and QA
    """

    def __init__(self, user_agent: str = "Mozilla/5.0", timeout_sec: int = 15):
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec

    def fetch_rss(self, feed_url: str, limit: int = 10) -> List[Dict[str, Optional[str]]]:
        req = urllib.request.Request(feed_url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            content = resp.read()

        feed = feedparser.parse(content)
        entries = []
        for e in feed.entries[:limit]:
            entries.append({
                "title": getattr(e, "title", None),
                "url": getattr(e, "link", None),
                "published": getattr(e, "published", None),
            })
        # drop invalid
        return [x for x in entries if x.get("url") and x.get("title")]

    def fetch_article_text(self, url: str) -> str:
        """
        Fetch and extract readable text from article URL.
        Uses trafilatura if available; falls back to raw HTML stripping (weaker).
        """
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            html = resp.read()

        if trafilatura is not None:
            downloaded = html.decode("utf-8", errors="ignore")
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text:
                return text

        # Fallback: very simple HTML tag removal (not ideal but works for Iteration 1)
        s = html.decode("utf-8", errors="ignore")
        s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
        s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
        s = re.sub(r"(?is)<.*?>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def normalize_text(self, text: str) -> str:
        """
        Normalization steps to clean up raw extracted text:
        - collapse whitespace/newlines
        - remove obvious boilerplate fragments (light)
        """
        t = text.replace("\u00a0", " ")
        t = re.sub(r"\s+", " ", t).strip()

        # light boilerplate removals (safe, minimal)
        t = re.sub(r"\bAdvertisement\b", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\bSign up\b.*$", "", t, flags=re.IGNORECASE).strip()

        # final whitespace cleanup
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def tokenize_stats(self, text: str) -> Dict[str, int]:
        """
        Tokenization stats for reporting/QA.
        Sentence tokenization is done in the model; here we keep quick counts.
        """
        # simple word count
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)

        # rough sentence count (period-based fallback)
        sentence_count = len([s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 0])

        return {"word_count": word_count, "sentence_count": sentence_count}

    def build_articles(self, feed_url: str, limit: int = 5) -> List[Article]:
        fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entries = self.fetch_rss(feed_url, limit=limit)

        articles: List[Article] = []
        for e in entries:
            url = e["url"]  # type: ignore[assignment]
            title = e["title"]  # type: ignore[assignment]
            published = e.get("published")

            raw = self.fetch_article_text(url)
            norm = self.normalize_text(raw)
            stats = self.tokenize_stats(norm)

            # skip too-short extraction
            if stats["word_count"] < 150:
                continue

            articles.append(
                Article(
                    title=title,
                    url=url,
                    published=published,
                    fetched_at=fetched_at,
                    raw_text=raw,
                    normalized_text=norm,
                    sentence_count=stats["sentence_count"],
                    word_count=stats["word_count"],
                )
            )

        return articles

    @staticmethod
    def to_dicts(articles: List[Article]) -> List[Dict]:
        return [asdict(a) for a in articles]