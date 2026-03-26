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

    Gracefully degrades if spaCy is not installed: verify() returns
    confidence=1.0 with empty entity lists.
    """

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

    def extract_entities(self, text: str) -> set[str]:
        """Extract normalized entity strings from text.

        Returns an empty set if spaCy is not available.
        """
        if self.nlp is None or not text.strip():
            return set()

        doc = self.nlp(text)
        # lowercase + strip so "Apple Inc." and "apple inc." match
        return {ent.text.strip().lower() for ent in doc.ents if ent.text.strip()}

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
