from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.abstractor import AbstractorBase, create_abstractor
from src.personas import format_prompt, get_persona
from src.summarizer_model import TextRankMMRSummarizer
from src.verifier import NERVerifier

VALID_MODES = {"extractive", "abstractive", "hybrid"}
VALID_LENGTHS = {"brief", "standard", "detailed"}


@dataclass
class PipelineResult:
    summary: str
    mode: str
    persona: str
    confidence: float = 1.0
    flagged_entities: list[str] = field(default_factory=list)
    extractive_sentences: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SummarizationPipeline:
    """Three-stage summarization pipeline: extract, abstract, verify."""

    def __init__(
        self,
        summarizer: TextRankMMRSummarizer | None = None,
        abstractor: AbstractorBase | None = None,
        verifier: NERVerifier | None = None,
    ) -> None:
        self.summarizer = summarizer or TextRankMMRSummarizer()
        self.abstractor = abstractor or create_abstractor()
        self.verifier = verifier or NERVerifier()

    def _extract(self, text: str, k: int) -> dict[str, Any]:
        return self.summarizer.summarize(text, k=k)

    def _abstract(
        self,
        sentences: list[str],
        persona_name: str,
        length: str,
    ) -> str:
        persona = get_persona(persona_name)
        user_prompt = format_prompt(persona, sentences, length)
        max_tokens = int(
            persona.max_tokens_hint
            * {"brief": 0.5, "standard": 1.0, "detailed": 2.0}.get(length, 1.0)
        )
        return self.abstractor.generate(
            system_prompt=persona.system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

    def run(
        self,
        text: str,
        mode: str = "extractive",
        persona: str = "default",
        length: str = "standard",
        k: int = 5,
    ) -> PipelineResult:
        if mode not in VALID_MODES:
            valid = ", ".join(sorted(VALID_MODES))
            raise ValueError(f"Unknown mode '{mode}'. Valid options: {valid}")

        if length not in VALID_LENGTHS:
            valid = ", ".join(sorted(VALID_LENGTHS))
            raise ValueError(f"Unknown length '{length}'. Valid options: {valid}")

        # Validate persona early
        get_persona(persona)

        if mode == "extractive":
            return self._run_extractive(text, persona, k)
        elif mode == "abstractive":
            return self._run_abstractive(text, persona, length, k)
        else:
            return self._run_hybrid(text, persona, length, k)

    def _run_extractive(
        self, text: str, persona: str, k: int
    ) -> PipelineResult:
        result = self._extract(text, k)
        selected = result.get("selected_indices", [])
        sentences = result.get("sentences", [])
        extracted = [sentences[i] for i in selected if i < len(sentences)]

        return PipelineResult(
            summary=str(result.get("summary", "")),
            mode="extractive",
            persona=persona,
            confidence=1.0,
            extractive_sentences=extracted,
        )

    def _run_abstractive(
        self, text: str, persona: str, length: str, k: int
    ) -> PipelineResult:
        # Use extraction to get key sentences, then abstract over them
        result = self._extract(text, k)
        selected = result.get("selected_indices", [])
        sentences = result.get("sentences", [])
        extracted = [sentences[i] for i in selected if i < len(sentences)]

        try:
            summary = self._abstract(extracted, persona, length)
        except Exception as exc:
            return PipelineResult(
                summary=str(result.get("summary", "")),
                mode="abstractive",
                persona=persona,
                confidence=1.0,
                extractive_sentences=extracted,
                metadata={"abstractor_error": str(exc), "fallback": "extractive"},
            )

        return PipelineResult(
            summary=summary,
            mode="abstractive",
            persona=persona,
            confidence=1.0,
            extractive_sentences=extracted,
        )

    def _run_hybrid(
        self, text: str, persona: str, length: str, k: int
    ) -> PipelineResult:
        # Stage 1: Extract
        result = self._extract(text, k)
        selected = result.get("selected_indices", [])
        sentences = result.get("sentences", [])
        extracted = [sentences[i] for i in selected if i < len(sentences)]

        # Stage 2: Abstract
        try:
            summary = self._abstract(extracted, persona, length)
        except Exception as exc:
            return PipelineResult(
                summary=str(result.get("summary", "")),
                mode="hybrid",
                persona=persona,
                confidence=1.0,
                extractive_sentences=extracted,
                metadata={"abstractor_error": str(exc), "fallback": "extractive"},
            )

        # Stage 3: Verify
        try:
            verification = self.verifier.verify(text, summary)
            confidence = verification.confidence
            flagged = verification.flagged_entities
        except Exception:
            confidence = 1.0
            flagged = []

        return PipelineResult(
            summary=summary,
            mode="hybrid",
            persona=persona,
            confidence=confidence,
            flagged_entities=flagged,
            extractive_sentences=extracted,
        )
