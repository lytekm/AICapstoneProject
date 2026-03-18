from __future__ import annotations

import re
import ssl
import urllib.request
from dataclasses import dataclass
from datetime import datetime

import feedparser

# Global SSL Fix for CNN/DailyMail certificates on Windows
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None

@dataclass
class Article:
    title: str
    url: str
    published: str | None
    fetched_at: str
    raw_text: str
    normalized_text: str
    sentence_count: int
    word_count: int

class NewsDataPipeline:
    def __init__(self, user_agent: str = "Mozilla/5.0", timeout_sec: int = 15):
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec

    def fetch_rss(self, feed_url: str, limit: int = 10) -> list[dict[str, str | None]]:
        try:
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
            return [x for x in entries if x.get("url") and x.get("title")]
        except Exception as e:
            print(f"Error fetching RSS {feed_url}: {e}")
            return []

    def fetch_article_text(self, url: str) -> str:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                html = resp.read()

            if trafilatura is not None:
                downloaded = html.decode("utf-8", errors="ignore")
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                if text:
                    return text

            s = html.decode("utf-8", errors="ignore")
            s = re.sub(r"(?is)<script.*?>.*?</script>|<style.*?>.*?</style>|<.*?>", " ", s)
            return re.sub(r"\s+", " ", s).strip()
        except Exception as e:
            print(f"Failed to extract {url}: {e}")
            return ""

    def normalize_text(self, text: str) -> str:
        # 1. Brute-force removal of common CBC navigation junk
        junk_header = r"Search Search Sign In Quick Links.*?Being Black in Canada More"
        text = re.sub(junk_header, "", text, flags=re.DOTALL)

        # 2. General cleaning
        t = text.replace("\u00a0", " ")
        patterns = [
            r"Advertisement", r"Sign up for our newsletter",
            r"Follow us on (Twitter|Facebook|Instagram|TikTok)",
            r"© \d{4} (CNN|CBC|Daily Mail|Associated Press)", r"All rights reserved"
        ]
        for p in patterns:
            t = re.sub(p, "", t, flags=re.IGNORECASE)

        t = re.sub(r"\s+", " ", t).strip()
        return t

    def tokenize_stats(self, text: str) -> dict[str, int]:
        words = re.findall(r"\b\w+\b", text)
        sentence_count = len([s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5])
        return {"word_count": len(words), "sentence_count": sentence_count}

    def build_articles(self, feed_url: str, limit: int = 5) -> list[Article]:
        fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entries = self.fetch_rss(feed_url, limit=limit)
        articles: list[Article] = []
        for e in entries:
            raw = self.fetch_article_text(e["url"])
            if not raw:
                continue
            norm = self.normalize_text(raw)
            stats = self.tokenize_stats(norm)
            if stats["word_count"] > 150:
                articles.append(Article(
                    title=e["title"], url=e["url"], published=e.get("published"),
                    fetched_at=fetched_at, raw_text=raw, normalized_text=norm,
                    sentence_count=stats["sentence_count"], word_count=stats["word_count"]
                ))
        return articles
