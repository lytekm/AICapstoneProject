"""Feedback collection and profile adjustment.

When a user likes or dislikes a summary, we record it and use
it to nudge their profile weights over time. This is what makes
the system actually learn from usage instead of just being a
static persona picker.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.user_profile import UserProfile


@dataclass
class FeedbackEntry:
    """A single like/dislike event on a summary."""

    user_id: str
    article_title: str
    persona: str
    mode: str
    liked: bool
    # ISO-8601 timestamp, filled automatically if not provided
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    """JSON file-backed feedback storage.

    Keeps a flat list of feedback entries per user. Same philosophy
    as ProfileStore: no external DB, professor can test on a laptop.
    """

    def __init__(self, path: str = "data/feedback.json") -> None:
        self._path = Path(path)
        self._entries: list[FeedbackEntry] = []
        self._load()

    def _load(self) -> None:
        """Read existing feedback from disk."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for item in raw:
                self._entries.append(FeedbackEntry(**item))
        except (json.JSONDecodeError, TypeError, KeyError):
            # corrupted file, start fresh
            self._entries = []

    def _flush(self) -> None:
        """Persist current entries to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out = [asdict(e) for e in self._entries]
        self._path.write_text(
            json.dumps(out, indent=2) + "\n",
            encoding="utf-8",
        )

    def record(self, entry: FeedbackEntry) -> None:
        """Add a feedback entry and save."""
        self._entries.append(entry)
        self._flush()

    def get_user_feedback(self, user_id: str) -> list[FeedbackEntry]:
        """Get all feedback entries for a specific user."""
        return [e for e in self._entries if e.user_id == user_id]

    def get_all(self) -> list[FeedbackEntry]:
        """Return every stored entry."""
        return list(self._entries)


# how much each like/dislike nudges the weight
_LIKE_DELTA = 0.1
_DISLIKE_DELTA = -0.05


def apply_feedback(
    profile: UserProfile,
    feedback: list[FeedbackEntry],
) -> UserProfile:
    """Adjust a user profile's feedback_weights based on their history.

    For each liked summary, we bump the weight of words in the title.
    For dislikes, we reduce it slightly. The asymmetry is intentional:
    we want to learn preferences faster than we penalize dislikes,
    since a dislike might just mean bad timing, not bad taste.
    """
    if not feedback:
        return profile

    # copy weights so we don't mutate the original
    weights = dict(profile.feedback_weights)

    for entry in feedback:
        # extract simple "topic" tokens from the article title
        # just split on spaces and take meaningful-length words
        tokens = [
            t.lower().strip(".,!?:;\"'()[]")
            for t in entry.article_title.split()
            if len(t) > 3  # skip short words like "the", "and", "a"
        ]

        delta = _LIKE_DELTA if entry.liked else _DISLIKE_DELTA
        for token in tokens:
            weights[token] = weights.get(token, 0.0) + delta

    # rebuild the profile with updated weights
    return UserProfile(
        user_id=profile.user_id,
        preferred_topics=profile.preferred_topics,
        keywords=profile.keywords,
        default_persona=profile.default_persona,
        default_length=profile.default_length,
        feedback_weights=weights,
    )
