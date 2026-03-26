"""Tests for feedback collection and profile weight adjustment."""

from __future__ import annotations

import json

import pytest

from src.feedback import FeedbackEntry, FeedbackStore, apply_feedback, _LIKE_DELTA, _DISLIKE_DELTA
from src.user_profile import UserProfile


# ---------------------------------------------------------------------------
# FeedbackEntry
# ---------------------------------------------------------------------------


class TestFeedbackEntry:
    def test_auto_timestamp(self) -> None:
        entry = FeedbackEntry(
            user_id="u1",
            article_title="AI is great",
            persona="casual",
            mode="hybrid",
            liked=True,
        )
        # should have a non-empty ISO timestamp
        assert entry.timestamp != ""
        assert "T" in entry.timestamp

    def test_explicit_timestamp(self) -> None:
        entry = FeedbackEntry(
            user_id="u1",
            article_title="test",
            persona="default",
            mode="extractive",
            liked=False,
            timestamp="2026-01-01T00:00:00",
        )
        assert entry.timestamp == "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# FeedbackStore
# ---------------------------------------------------------------------------


class TestFeedbackStore:
    def _make_store(self, tmp_path: str) -> FeedbackStore:
        return FeedbackStore(path=f"{tmp_path}/feedback.json")

    def _entry(self, user_id: str = "u1", title: str = "AI News", liked: bool = True) -> FeedbackEntry:
        return FeedbackEntry(
            user_id=user_id,
            article_title=title,
            persona="default",
            mode="extractive",
            liked=liked,
        )

    def test_empty_store(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        assert store.get_all() == []
        assert store.get_user_feedback("nobody") == []

    def test_record_and_retrieve(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        store.record(self._entry("u1", "Article A"))
        store.record(self._entry("u1", "Article B"))
        store.record(self._entry("u2", "Article C"))

        u1_fb = store.get_user_feedback("u1")
        assert len(u1_fb) == 2
        assert u1_fb[0].article_title == "Article A"

        u2_fb = store.get_user_feedback("u2")
        assert len(u2_fb) == 1

    def test_persistence_across_instances(self, tmp_path: str) -> None:
        path = f"{tmp_path}/feedback.json"
        store1 = FeedbackStore(path=path)
        store1.record(self._entry("u1", "Persistent Article"))

        store2 = FeedbackStore(path=path)
        assert len(store2.get_user_feedback("u1")) == 1
        assert store2.get_user_feedback("u1")[0].article_title == "Persistent Article"

    def test_corrupted_file_recovers(self, tmp_path: str) -> None:
        path = f"{tmp_path}/feedback.json"
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text("{{{NOT JSON", encoding="utf-8")

        store = FeedbackStore(path=path)
        assert store.get_all() == []

    def test_get_all(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        store.record(self._entry("u1"))
        store.record(self._entry("u2"))
        assert len(store.get_all()) == 2


# ---------------------------------------------------------------------------
# apply_feedback
# ---------------------------------------------------------------------------


class TestApplyFeedback:
    def test_empty_feedback_returns_unchanged(self) -> None:
        profile = UserProfile(user_id="u1", preferred_topics=["AI"])
        result = apply_feedback(profile, [])
        assert result.preferred_topics == ["AI"]
        assert result.feedback_weights == {}

    def test_like_increases_weight(self) -> None:
        profile = UserProfile(user_id="u1")
        feedback = [
            FeedbackEntry(
                user_id="u1",
                article_title="Machine Learning Breakthrough",
                persona="default",
                mode="extractive",
                liked=True,
            ),
        ]
        result = apply_feedback(profile, feedback)
        # "machine", "learning", "breakthrough" should all get bumped
        assert result.feedback_weights.get("machine", 0.0) == pytest.approx(_LIKE_DELTA)
        assert result.feedback_weights.get("learning", 0.0) == pytest.approx(_LIKE_DELTA)
        assert result.feedback_weights.get("breakthrough", 0.0) == pytest.approx(_LIKE_DELTA)

    def test_dislike_decreases_weight(self) -> None:
        profile = UserProfile(user_id="u1")
        feedback = [
            FeedbackEntry(
                user_id="u1",
                article_title="Boring Stock Market Update",
                persona="default",
                mode="extractive",
                liked=False,
            ),
        ]
        result = apply_feedback(profile, feedback)
        assert result.feedback_weights.get("boring", 0.0) == pytest.approx(_DISLIKE_DELTA)
        assert result.feedback_weights.get("stock", 0.0) == pytest.approx(_DISLIKE_DELTA)

    def test_short_words_skipped(self) -> None:
        """Words with 3 or fewer characters should be ignored."""
        profile = UserProfile(user_id="u1")
        feedback = [
            FeedbackEntry(
                user_id="u1",
                article_title="AI is the new oil",
                persona="default",
                mode="extractive",
                liked=True,
            ),
        ]
        result = apply_feedback(profile, feedback)
        # "AI", "is", "the", "new", "oil" are all <= 3 chars, none should appear
        assert len(result.feedback_weights) == 0

    def test_does_not_mutate_original(self) -> None:
        profile = UserProfile(user_id="u1", feedback_weights={"tech": 0.5})
        feedback = [
            FeedbackEntry(
                user_id="u1",
                article_title="Tech Giants Report Earnings",
                persona="default",
                mode="extractive",
                liked=True,
            ),
        ]
        result = apply_feedback(profile, feedback)
        # original should be untouched
        assert profile.feedback_weights == {"tech": 0.5}
        # result should have the adjustment
        assert result.feedback_weights.get("tech", 0.0) == pytest.approx(0.5 + _LIKE_DELTA)

    def test_multiple_feedback_accumulates(self) -> None:
        profile = UserProfile(user_id="u1")
        feedback = [
            FeedbackEntry(
                user_id="u1",
                article_title="NVIDIA Launches New Chip",
                persona="default",
                mode="extractive",
                liked=True,
            ),
            FeedbackEntry(
                user_id="u1",
                article_title="NVIDIA Stock Surges After Launch",
                persona="default",
                mode="extractive",
                liked=True,
            ),
        ]
        result = apply_feedback(profile, feedback)
        # "nvidia" appears in both, should get 2x the like delta
        assert result.feedback_weights.get("nvidia", 0.0) == pytest.approx(2 * _LIKE_DELTA)
