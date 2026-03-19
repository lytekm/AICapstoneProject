"""
FastAPI Backend: AI Capstone Project
Provides endpoints to fetch RSS articles and summarize selected articles
with extractive, abstractive, or hybrid modes and persona support.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Iterator
from typing import Any

import feedparser
import trafilatura
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.personas import PERSONAS
from src.pipeline import SummarizationPipeline

# ----------------------------
# App setup
# ----------------------------
app = FastAPI(title="AI Capstone Summarizer API", version="2.0")

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

pipeline = SummarizationPipeline()


# ----------------------------
# Request / Response models
# ----------------------------
class SummarizeRequest(BaseModel):
    url: str
    k: int = Field(default=DEFAULT_K, ge=1, le=20)
    mode: str = "extractive"
    persona: str = "default"
    length: str = "standard"


class SummarizeResponse(BaseModel):
    summary: str
    mode: str
    persona: str
    confidence: float | None = None
    flagged_entities: list[str] | None = None


# ----------------------------
# Helpers
# ----------------------------
def _fetch_url(url: str, timeout: int = REQUEST_TIMEOUT_SEC) -> str:
    """Download a URL and return HTML as string."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html: str = resp.read().decode("utf-8", errors="ignore")
            return html
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


@app.get("/api/personas")
def list_personas() -> dict[str, list[str]]:
    """Return available persona names."""
    return {"personas": sorted(PERSONAS.keys())}


@app.post("/api/summarize")
def summarize(data: dict[str, Any]) -> dict[str, Any]:
    """Summarize an article with optional mode, persona, and length control.

    Request body:
    {
      "url": "https://example.com/article",
      "k": 5,
      "mode": "extractive",      // extractive | abstractive | hybrid
      "persona": "default",      // technical | casual | executive | academic | default
      "length": "standard"       // brief | standard | detailed
    }
    """
    url = (data.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body.")

    mode = str(data.get("mode", "extractive")).strip().lower()
    persona = str(data.get("persona", "default")).strip().lower()
    length = str(data.get("length", "standard")).strip().lower()
    k = _parse_k(data.get("k", DEFAULT_K))

    # Validate persona early
    if persona not in PERSONAS:
        valid = ", ".join(sorted(PERSONAS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona '{persona}'. Valid options: {valid}",
        )

    # Download and extract
    html = _fetch_url(url)
    text = _extract_main_text(html)

    try:
        result = pipeline.run(
            text=text, mode=mode, persona=persona, length=length, k=k
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e

    if not result.summary.strip():
        raise HTTPException(status_code=500, detail="Pipeline returned an empty summary.")

    response: dict[str, Any] = {
        "summary": result.summary,
        "mode": result.mode,
        "persona": result.persona,
    }

    if mode in ("hybrid", "abstractive"):
        response["confidence"] = result.confidence
        response["flagged_entities"] = result.flagged_entities

    return response


@app.get("/api/summarize/stream")
def summarize_stream(
    url: str = Query(..., description="Article URL to summarize"),
    k: int = Query(DEFAULT_K, ge=1, le=20),
    mode: str = Query("hybrid", description="extractive | abstractive | hybrid"),
    persona: str = Query("default"),
    length: str = Query("standard", description="brief | standard | detailed"),
) -> StreamingResponse:
    """Stream a summary via Server-Sent Events (SSE).

    Uses GET so the browser's native EventSource API can connect directly.
    Tokens arrive as `event: token` messages; the final result comes as
    `event: done` with confidence and flagged entities (for hybrid mode).
    """
    url = url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' query parameter.")

    mode = mode.strip().lower()
    persona = persona.strip().lower()
    length = length.strip().lower()

    # validate persona before we start streaming
    if persona not in PERSONAS:
        valid = ", ".join(sorted(PERSONAS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown persona '{persona}'. Valid options: {valid}",
        )

    # download and extract article text up front
    html = _fetch_url(url)
    text = _extract_main_text(html)

    def _event_generator() -> Iterator[str]:
        yield from pipeline.run_stream(
            text=text, mode=mode, persona=persona, length=length, k=k, delay=0.0
        )

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
