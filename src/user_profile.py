"""User profile model and JSON-backed persistence.

Each user can store their preferred topics, keywords, and defaults
so the system actually adapts to them over time instead of treating
every request as a blank slate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UserProfile:
    """Represents a single user's summarization preferences."""

    user_id: str
    # topics the user cares about, used for article ranking
    preferred_topics: list[str] = field(default_factory=list)
    # specific words to boost when ranking articles
    keywords: list[str] = field(default_factory=list)
    # persona and length the user prefers by default
    default_persona: str = "default"
    default_length: str = "standard"
    # adjusted by the feedback loop: topic -> weight delta
    # positive means the user tends to like articles on that topic
    feedback_weights: dict[str, float] = field(default_factory=dict)


def _profile_from_dict(data: dict[str, Any]) -> UserProfile:
    """Build a UserProfile from a raw dict (loaded from JSON)."""
    return UserProfile(
        user_id=data["user_id"],
        preferred_topics=data.get("preferred_topics", []),
        keywords=data.get("keywords", []),
        default_persona=data.get("default_persona", "default"),
        default_length=data.get("default_length", "standard"),
        feedback_weights=data.get("feedback_weights", {}),
    )


class ProfileStore:
    """JSON file-backed user profile storage.

    No external database needed. The professor can test this
    on a laptop with zero setup -- profiles just live in a
    JSON file that gets created on first save.
    """

    def __init__(self, path: str = "data/profiles.json") -> None:
        self._path = Path(path)
        # keep everything in memory, flush to disk on writes
        self._profiles: dict[str, UserProfile] = {}
        self._load()

    def _load(self) -> None:
        """Read profiles from disk if the file exists."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            # the file stores a dict keyed by user_id, but the inner
            # dicts don't redundantly store user_id -- we inject it here
            for uid, data in raw.items():
                data["user_id"] = uid
                self._profiles[uid] = _profile_from_dict(data)
        except (json.JSONDecodeError, KeyError):
            # corrupted file -- start fresh rather than crash
            # better to lose saved profiles than to error on every request
            self._profiles = {}

    def _flush(self) -> None:
        """Write current state to disk."""
        # make sure the parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out = {uid: asdict(p) for uid, p in self._profiles.items()}
        self._path.write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def get(self, user_id: str) -> UserProfile | None:
        """Look up a profile by user_id. Returns None if not found."""
        return self._profiles.get(user_id)

    def save(self, profile: UserProfile) -> None:
        """Create or update a user profile and persist to disk."""
        self._profiles[profile.user_id] = profile
        self._flush()

    def delete(self, user_id: str) -> bool:
        """Remove a profile. Returns True if it existed."""
        if user_id in self._profiles:
            del self._profiles[user_id]
            self._flush()
            return True
        return False

    def list_users(self) -> list[str]:
        """Return all stored user IDs."""
        return sorted(self._profiles.keys())
