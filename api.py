"""
FastAPI Backend: AI Capstone Project - Iteration 1
Provides endpoints to fetch RSS articles and summarize selected articles.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any

import feedparser
import trafilatura
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.summarizer_model import SummarizerConfig, TextRankMMRSummarizer

# ----------------------------
# App setup
# ----------------------------
app = FastAPI(title="AI Capstone Summarizer API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_RSS = os.getenv("RSS_FEED_URL", "https://www.cbc.ca/cmlink/rss-business")
DEFAULT_K = 5
MIN_TEXT_CHARS = 200
REQUEST_TIMEOUT_SEC = 20


# ----------------------------
# Helpers
# ----------------------------
def _fetch_url(url: str, timeout: int = REQUEST_TIMEOUT_SEC) -> str:
    """Download a URL and return HTML as string."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download article: {e}") from e


def _extract_main_text(html: str) -> str:
    """Extract main readable article text from HTML."""
    try:
        text = trafilatura.extract(html) or ""
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Extraction failed: {e}") from e

    if len(text.strip()) < MIN_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail="Could not extract enough article text from this URL.",
        )
    return text


def _parse_k(value: Any, default: int = DEFAULT_K, k_min: int = 1, k_max: int = 20) -> int:
    """Parse and clamp summary length k."""
    try:
        k = int(value) if value is not None else int(default)
    except Exception:
        k = int(default)
    return max(k_min, min(k, k_max))


# ----------------------------
# Routes
# ----------------------------
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/articles")
def get_articles() -> list[dict[str, str]]:
    """Fetch top articles from the default RSS feed."""
    try:
        request = urllib.request.Request(DEFAULT_RSS, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
            feed = feedparser.parse(response.read())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch RSS feed: {e}") from e

    return [{"title": e.title, "link": e.link} for e in feed.entries[:10]]


@app.post("/api/summarize")
def summarize(data: dict[str, Any]) -> dict[str, str]:
    """
    Request body example:
    {
      "url": "https://example.com/article",
      "k": 5
    }
    """
    url = (data.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body.")

    # 1) Download HTML
    html = _fetch_url(url)

    # 2) Extract main text
    text = _extract_main_text(html)

    # 3) Summarize using user-provided k
    try:
        config = SummarizerConfig(mmr_lambda=0.75, blend_alpha=0.7)
        summarizer = TextRankMMRSummarizer(config)

        k = _parse_k(data.get("k", DEFAULT_K))
        result = summarizer.summarize(text, k=k)

        summary = result.get("summary", "")
        if not summary.strip():
            raise HTTPException(status_code=500, detail="Summarizer returned an empty summary.")

        return {"summary": summary}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarizer crashed: {e}") from e

