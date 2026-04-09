"""NER-based factual consistency verification.

After the LLM rewrites a summary, we need to check whether it
hallucinated any named entities (people, organizations, locations, etc.)
that weren't in the original article. This module does that by
comparing spaCy NER output between source and summary.

If an entity shows up in the summary but not the source, it's
flagged as potentially hallucinated. The confidence score drops
proportionally to how many entities were flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationResult:
    """Output of the NER consistency check."""

    confidence: float
    # entities in the summary that don't appear in the source
    flagged_entities: list[str] = field(default_factory=list)
    # all entities found in the source (for debugging/display)
    source_entities: list[str] = field(default_factory=list)
    summary_entities: list[str] = field(default_factory=list)


class NERVerifier:
    """NER-based factual consistency checker using spaCy.

    Compares named entities in a generated summary against the source text.
    Entities present in the summary but absent from the source are flagged,
    and confidence is reduced proportionally.

    Gracefully degrades if spaCy is not installed: the pipeline will skip
    verification. Direct verify() calls still return a benign result so the
    verifier remains safe to use in isolation.
    """

    IGNORED_LABELS = {
        "CARDINAL",
        "DATE",
        "MONEY",
        "ORDINAL",
        "PERCENT",
        "QUANTITY",
        "TIME",
    }
    IGNORED_ATTRIBUTION_ENTITIES = {
        "ap",
        "associated press",
        "bloomberg",
        "cnn business",
        "reuters",
        "yahoo finance",
    }
    PARENTHETICAL_CITATION_RE = re.compile(
        r"\((?:[^()]*?,\s*)?(?:19\d{2}|20\d{2}|20XX)(?:[a-z])?[^()]*\)"
    )
    REFERENCES_SECTION_RE = re.compile(r"(?is)\breferences\s*:.*$")
    RETRIEVED_FROM_RE = re.compile(r"(?im)^\s*retrieved from\s+.*$")
    BULLET_PREFIX_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")
    MARKDOWN_RE = re.compile(r"[*_`#]+")
    URL_RE = re.compile(r"https?://\S+")
    WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.nlp: Any = None
        try:
            import spacy

            self.nlp = spacy.load(model_name)
        except Exception:
            # spaCy not installed or model not downloaded -- that's ok,
            # we just skip verification and return full confidence
            pass

    @property
    def available(self) -> bool:
        return self.nlp is not None

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Strip common formatting/citation noise before running NER."""
        if not text.strip():
            return ""

        cleaned = cls.MARKDOWN_RE.sub("", text)
        cleaned = cls.REFERENCES_SECTION_RE.sub("", cleaned)
        cleaned = cls.RETRIEVED_FROM_RE.sub("", cleaned)
        cleaned = cls.PARENTHETICAL_CITATION_RE.sub("", cleaned)
        cleaned = cls.BULLET_PREFIX_RE.sub("", cleaned)
        cleaned = cls.URL_RE.sub("", cleaned)
        return cls.WHITESPACE_RE.sub(" ", cleaned).strip()

    @staticmethod
    def normalize_entity(text: str) -> str:
        """Lowercase and trim punctuation so minor formatting differences match."""
        return re.sub(r"\s+", " ", text.strip().strip("()[]{}.,:;\"'")).lower()

    @classmethod
    def should_ignore_entity(cls, text: str, label: str) -> bool:
        """Skip entity classes that create noisy, low-value hallucination alerts."""
        normalized = cls.normalize_entity(text)
        if not normalized:
            return True
        if label in cls.IGNORED_LABELS:
            return True
        if normalized in cls.IGNORED_ATTRIBUTION_ENTITIES:
            return True
        return re.fullmatch(r"\d+(?:[./-]\d+)*(?:%|st|nd|rd|th)?", normalized) is not None

    def extract_entities(self, text: str) -> set[str]:
        """Extract normalized entity strings from text.

        Returns an empty set if spaCy is not available.
        """
        cleaned_text = self.sanitize_text(text)
        if self.nlp is None or not cleaned_text:
            return set()

        doc = self.nlp(cleaned_text)
        entities: set[str] = set()
        for ent in doc.ents:
            normalized = self.normalize_entity(ent.text)
            if self.should_ignore_entity(normalized, ent.label_):
                continue
            entities.add(normalized)
        return entities

    def verify(self, source_text: str, summary_text: str) -> VerificationResult:
        """Compare entities between source and summary.

        Confidence = 1.0 - (flagged / total_summary_entities).
        If spaCy is unavailable or summary has no entities, confidence = 1.0.
        """
        if self.nlp is None:
            return VerificationResult(confidence=1.0)

        source_ents = self.extract_entities(source_text)
        summary_ents = self.extract_entities(summary_text)

        # no entities in the summary means nothing to verify
        if not summary_ents:
            return VerificationResult(
                confidence=1.0,
                source_entities=sorted(source_ents),
                summary_entities=[],
            )

        # set difference: entities in summary but NOT in source = suspicious
        flagged = summary_ents - source_ents
        # more flagged entities = lower confidence
        confidence = 1.0 - (len(flagged) / len(summary_ents))

        return VerificationResult(
            # floor at 0.0 in case everything got flagged
            confidence=max(0.0, confidence),
            flagged_entities=sorted(flagged),
            source_entities=sorted(source_ents),
            summary_entities=sorted(summary_ents),
        )
