"""
FastAPI Backend: AI Capstone Project
Provides endpoints to fetch RSS articles and summarize selected articles
with extractive, abstractive, or hybrid modes and persona support.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import feedparser
import trafilatura
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictBool

from src.article_ranker import ArticleRanker
from src.feedback import FeedbackEntry, FeedbackStore, apply_feedback
from src.personas import PERSONAS
from src.pipeline import SummarizationPipeline
from src.user_profile import ProfileStore, UserProfile

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
profile_store = ProfileStore()
feedback_store = FeedbackStore()
article_ranker = ArticleRanker()


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
    flagged_entities: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    user_id: str
    article_title: str
    persona: str = "default"
    mode: str = "extractive"
    liked: StrictBool


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


def _build_summarize_response(result: Any, mode: str) -> dict[str, Any]:
    """Normalize summarize responses so all modes share the same wire shape."""
    confidence = (
        result.confidence
        if mode == "hybrid" and result.confidence is not None
        else None
    )
    flagged_entities = (
        list(result.flagged_entities)
        if mode == "hybrid" and result.confidence is not None
        else []
    )
    return cast(dict[str, Any], SummarizeResponse(
        summary=result.summary,
        mode=result.mode,
        persona=result.persona,
        confidence=confidence,
        flagged_entities=flagged_entities,
    ).model_dump())


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

    # if a user_id is provided, load their profile defaults
    # explicit params in the request always override profile defaults
    user_id = (data.get("user_id") or "").strip()
    profile = profile_store.get(user_id) if user_id else None

    mode = str(data.get("mode", "extractive")).strip().lower()
    persona = str(data.get("persona") or (profile.default_persona if profile else "default")).strip().lower()
    length = str(data.get("length") or (profile.default_length if profile else "standard")).strip().lower()
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

    return _build_summarize_response(result, mode)


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


# ----------------------------
# User profile endpoints
# ----------------------------
@app.post("/api/user/preferences")
def save_user_preferences(data: dict[str, Any]) -> dict[str, Any]:
    """Create or update a user profile.

    Expects: {"user_id": "...", "preferred_topics": [...], "keywords": [...],
              "default_persona": "...", "default_length": "..."}
    """
    user_id = (data.get("user_id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing 'user_id'.")

    # validate persona if provided
    persona = str(data.get("default_persona", "default")).strip().lower()
    if persona not in PERSONAS:
        valid = ", ".join(sorted(PERSONAS.keys()))
        raise HTTPException(status_code=400, detail=f"Unknown persona '{persona}'. Valid: {valid}")

    profile = UserProfile(
        user_id=user_id,
        preferred_topics=data.get("preferred_topics", []),
        keywords=data.get("keywords", []),
        default_persona=persona,
        default_length=str(data.get("default_length", "standard")).strip().lower(),
    )

    # carry over existing feedback weights if the user already had a profile
    existing = profile_store.get(user_id)
    if existing:
        profile.feedback_weights = existing.feedback_weights

    profile_store.save(profile)
    return {"status": "saved", "user_id": user_id}


@app.get("/api/user/preferences/{user_id}")
def get_user_preferences(user_id: str) -> dict[str, Any]:
    """Retrieve a user's stored preferences."""
    profile = profile_store.get(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile found for '{user_id}'.")
    return {
        "user_id": profile.user_id,
        "preferred_topics": profile.preferred_topics,
        "keywords": profile.keywords,
        "default_persona": profile.default_persona,
        "default_length": profile.default_length,
        "feedback_weights": profile.feedback_weights,
    }


@app.post("/api/user/feedback")
def record_feedback(data: FeedbackRequest) -> dict[str, str]:
    """Record a like or dislike on a summary.

    Expects: {"user_id": "...", "article_title": "...", "persona": "...",
              "mode": "...", "liked": true/false}
    """
    user_id = data.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing 'user_id'.")

    article_title = data.article_title.strip()
    if not article_title:
        raise HTTPException(status_code=400, detail="Missing 'article_title'.")

    entry = FeedbackEntry(
        user_id=user_id,
        article_title=article_title,
        persona=data.persona.strip(),
        mode=data.mode.strip(),
        liked=data.liked,
    )
    feedback_store.record(entry)

    # apply feedback to the user's profile if they have one
    profile = profile_store.get(user_id)
    if profile:
        updated = apply_feedback(profile, [entry])
        profile_store.save(updated)

    return {"status": "recorded"}


@app.get("/api/articles/personalized")
def get_personalized_articles(
    user_id: str = Query(..., description="User ID for personalized ranking"),
) -> list[dict[str, Any]]:
    """Fetch and rank articles based on a user's preferences."""
    profile = profile_store.get(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile found for '{user_id}'.")

    # fetch articles the same way as the regular endpoint
    try:
        request = urllib.request.Request(DEFAULT_RSS, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
            feed = feedparser.parse(response.read())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch RSS feed: {e}") from e

    articles = [{"title": e.title, "link": e.link} for e in feed.entries[:10]]
    ranked = article_ranker.rank(articles, profile)

    return [
        {
            "title": r.title,
            "link": r.link,
            "score": r.score,
            "match_reasons": r.match_reasons,
        }
        for r in ranked
    ]


# ----------------------------
# Static frontend serving
# ----------------------------
# SvelteKit builds static files into frontend/. We serve them here so the
# same uvicorn process handles both the API and the UI -- no separate web
# server needed for local dev or single-container deployment.

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

if FRONTEND_DIR.is_dir():
    # serve the SvelteKit _app directory (JS/CSS bundles)
    app.mount("/_app", StaticFiles(directory=FRONTEND_DIR / "_app"), name="svelte-app")

    # catch-all: any non-API route returns index.html so SvelteKit client
    # routing works (e.g. /summarize, /profile, /compare).
    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str) -> FileResponse:
        # let unknown API paths return a real API 404 instead of HTML
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        # try to serve a static file first (e.g. favicon.png)
        file_path = FRONTEND_DIR / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        # fall back to index.html for client-side routing
        return FileResponse(FRONTEND_DIR / "index.html")
