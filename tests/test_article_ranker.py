"""Tests for article ranking based on user profile preferences."""

from __future__ import annotations

import pytest

from src.article_ranker import ArticleRanker, RankedArticle, _feedback_score, _keyword_score, _topic_score
from src.user_profile import UserProfile


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


class TestTopicScore:
    def test_no_topics(self) -> None:
        score, reasons = _topic_score("some headline", [])
        assert score == 0.0
        assert reasons == []

    def test_matching_topic(self) -> None:
        score, reasons = _topic_score("new ai model released", ["AI"])
        assert score == pytest.approx(1.0)
        assert "topic: AI" in reasons

    def test_partial_match(self) -> None:
        score, reasons = _topic_score("ai is growing", ["AI", "finance"])
        assert score == pytest.approx(0.5)
        assert len(reasons) == 1

    def test_no_match(self) -> None:
        score, reasons = _topic_score("sports update today", ["AI", "finance"])
        assert score == 0.0
        assert reasons == []


class TestKeywordScore:
    def test_no_keywords(self) -> None:
        score, reasons = _keyword_score("headline", [])
        assert score == 0.0

    def test_keyword_found(self) -> None:
        score, reasons = _keyword_score("nvidia releases new gpu", ["GPU"])
        assert score == pytest.approx(1.0)
        assert "keyword: GPU" in reasons

    def test_keyword_not_found(self) -> None:
        score, reasons = _keyword_score("apple earnings report", ["GPU"])
        assert score == 0.0


class TestFeedbackScore:
    def test_no_weights(self) -> None:
        assert _feedback_score("headline", {}) == 0.0

    def test_positive_weight(self) -> None:
        score = _feedback_score("ai breakthrough", {"ai": 0.8})
        assert score > 0.0

    def test_no_match(self) -> None:
        score = _feedback_score("sports news", {"ai": 0.8})
        assert score == 0.0

    def test_clamped_to_one(self) -> None:
        # even a huge weight should cap at 1.0
        score = _feedback_score("ai news", {"ai": 5.0})
        assert score <= 1.0


# ---------------------------------------------------------------------------
# ArticleRanker
# ---------------------------------------------------------------------------


SAMPLE_ARTICLES = [
    {"title": "AI Startups Raise Record Funding", "link": "https://example.com/1"},
    {"title": "Bank of Canada Holds Interest Rate", "link": "https://example.com/2"},
    {"title": "New GPU Architecture from NVIDIA", "link": "https://example.com/3"},
    {"title": "Sports: Raptors Win Season Opener", "link": "https://example.com/4"},
]


class TestArticleRanker:
    def setup_method(self) -> None:
        self.ranker = ArticleRanker()

    def test_no_profile_preserves_order(self) -> None:
        """Without a profile, articles come back in original order with score 0."""
        result = self.ranker.rank(SAMPLE_ARTICLES, profile=None)
        assert len(result) == 4
        assert all(r.score == 0.0 for r in result)
        assert result[0].title == "AI Startups Raise Record Funding"

    def test_topic_ranking(self) -> None:
        profile = UserProfile(user_id="u1", preferred_topics=["AI"])
        result = self.ranker.rank(SAMPLE_ARTICLES, profile=profile)
        # AI article should be first
        assert result[0].title == "AI Startups Raise Record Funding"
        assert result[0].score > 0.0
        assert any("topic: AI" in r for r in result[0].match_reasons)

    def test_keyword_ranking(self) -> None:
        profile = UserProfile(user_id="u1", keywords=["GPU"])
        result = self.ranker.rank(SAMPLE_ARTICLES, profile=profile)
        # GPU article should rank high
        gpu_article = next(r for r in result if "GPU" in r.title)
        assert gpu_article.score > 0.0

    def test_combined_signals(self) -> None:
        """An article matching both topic and keyword should score highest."""
        profile = UserProfile(
            user_id="u1",
            preferred_topics=["AI"],
            keywords=["startup"],
        )
        result = self.ranker.rank(SAMPLE_ARTICLES, profile=profile)
        # "AI Startups..." matches both topic and keyword
        assert result[0].title == "AI Startups Raise Record Funding"
        # should score higher than a topic-only or keyword-only match
        assert result[0].score > result[1].score

    def test_feedback_weights_boost(self) -> None:
        """Feedback weights should influence ranking."""
        profile = UserProfile(
            user_id="u1",
            feedback_weights={"bank": 0.9, "interest": 0.9},
        )
        result = self.ranker.rank(SAMPLE_ARTICLES, profile=profile)
        # the finance article should get boosted by feedback
        bank_article = next(r for r in result if "Bank" in r.title)
        assert bank_article.score > 0.0

    def test_empty_articles(self) -> None:
        result = self.ranker.rank([], profile=None)
        assert result == []

    def test_empty_profile_preferences(self) -> None:
        """A profile with no topics/keywords returns original order."""
        profile = UserProfile(user_id="u1")
        result = self.ranker.rank(SAMPLE_ARTICLES, profile=profile)
        assert all(r.score == 0.0 for r in result)

    def test_result_is_ranked_article(self) -> None:
        result = self.ranker.rank(SAMPLE_ARTICLES, profile=None)
        assert all(isinstance(r, RankedArticle) for r in result)

    def test_stable_sort_on_ties(self) -> None:
        """Articles with the same score should keep their original order."""
        profile = UserProfile(user_id="u1")
        result = self.ranker.rank(SAMPLE_ARTICLES, profile=profile)
        # all scores are 0, so original order should be preserved
        assert result[0].title == SAMPLE_ARTICLES[0]["title"]
        assert result[3].title == SAMPLE_ARTICLES[3]["title"]
