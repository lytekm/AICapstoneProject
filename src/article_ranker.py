"""Article ranking based on user profile preferences.

Scores each article against a user's preferred topics, keywords, and
feedback history so the most relevant articles float to the top.
Without a profile, articles come back in their original order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.user_profile import UserProfile


@dataclass
class RankedArticle:
    """An article with a relevance score and reasons for that score."""

    title: str
    link: str
    score: float = 0.0
    # human-readable reasons so the frontend can show why this ranked high
    match_reasons: list[str] = field(default_factory=list)


# how much each signal contributes to the final score
_TOPIC_WEIGHT = 0.5
_KEYWORD_WEIGHT = 0.3
_FEEDBACK_WEIGHT = 0.2


def _topic_score(title_lower: str, topics: list[str]) -> tuple[float, list[str]]:
    """Check if any preferred topics appear in the article title.

    Returns a 0-1 score and a list of matched topic reasons.
    Simple substring matching -- good enough for news headlines.
    """
    if not topics:
        return 0.0, []

    matched = []
    for topic in topics:
        # case-insensitive substring check against the title
        if topic.lower() in title_lower:
            matched.append(f"topic: {topic}")

    # proportion of topics that matched
    score = len(matched) / len(topics) if topics else 0.0
    return score, matched


def _keyword_score(title_lower: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Check if any user keywords appear in the article title."""
    if not keywords:
        return 0.0, []

    matched = []
    for kw in keywords:
        if kw.lower() in title_lower:
            matched.append(f"keyword: {kw}")

    score = len(matched) / len(keywords) if keywords else 0.0
    return score, matched


def _feedback_score(title_lower: str, weights: dict[str, float]) -> float:
    """Apply feedback weights: topics the user liked get a boost."""
    if not weights:
        return 0.0

    total = 0.0
    hits = 0
    for topic, weight in weights.items():
        if topic.lower() in title_lower:
            total += weight
            hits += 1

    # normalize so one really strong weight doesn't dominate
    if hits == 0:
        return 0.0
    # clamp between 0 and 1
    return max(0.0, min(1.0, total / hits))


class ArticleRanker:
    """Ranks a list of articles by relevance to a user profile."""

    def rank(
        self,
        articles: list[dict[str, str]],
        profile: UserProfile | None = None,
    ) -> list[RankedArticle]:
        """Score and sort articles by relevance.

        Each article dict should have at least 'title' and 'link' keys.
        If no profile is provided, articles keep their original order
        with a score of 0.0 (no ranking signal).
        """
        if profile is None:
            # no profile means no ranking, just wrap them
            return [
                RankedArticle(title=a.get("title", ""), link=a.get("link", ""))
                for a in articles
            ]

        ranked: list[RankedArticle] = []
        for article in articles:
            title = article.get("title", "")
            link = article.get("link", "")
            title_lower = title.lower()

            reasons: list[str] = []

            t_score, t_reasons = _topic_score(title_lower, profile.preferred_topics)
            reasons.extend(t_reasons)

            k_score, k_reasons = _keyword_score(title_lower, profile.keywords)
            reasons.extend(k_reasons)

            f_score = _feedback_score(title_lower, profile.feedback_weights)

            # weighted combination of all signals
            final = (
                _TOPIC_WEIGHT * t_score
                + _KEYWORD_WEIGHT * k_score
                + _FEEDBACK_WEIGHT * f_score
            )

            ranked.append(RankedArticle(
                title=title,
                link=link,
                score=round(final, 4),
                match_reasons=reasons,
            ))

        # highest score first, stable sort preserves original order for ties
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked
