from __future__ import annotations

import json
from collections.abc import Iterator
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

    # -- streaming support --

    def run_stream(
        self,
        text: str,
        mode: str = "extractive",
        persona: str = "default",
        length: str = "standard",
        k: int = 5,
        delay: float = 0.02,
    ) -> Iterator[str]:
        """Yield SSE-formatted events for streaming responses."""
        if mode not in VALID_MODES:
            yield _sse("error", {"detail": f"Unknown mode '{mode}'"})
            return
        if length not in VALID_LENGTHS:
            yield _sse("error", {"detail": f"Unknown length '{length}'"})
            return
        try:
            get_persona(persona)
        except ValueError as exc:
            yield _sse("error", {"detail": str(exc)})
            return

        # stage 1: extract (always needed, even for abstractive)
        result = self._extract(text, k)
        selected = result.get("selected_indices", [])
        sentences = result.get("sentences", [])
        extracted = [sentences[i] for i in selected if i < len(sentences)]

        # for extractive mode, just send the whole summary at once
        if mode == "extractive":
            yield _sse("done", {
                "summary": str(result.get("summary", "")),
                "mode": "extractive",
                "persona": persona,
                "confidence": 1.0,
                "flagged_entities": [],
            })
            return

        # send metadata before streaming starts
        yield _sse("meta", {"mode": mode, "persona": persona})

        # stage 2: stream the abstraction token by token
        persona_obj = get_persona(persona)
        user_prompt = format_prompt(persona_obj, extracted, length)
        max_tokens = int(
            persona_obj.max_tokens_hint
            * {"brief": 0.5, "standard": 1.0, "detailed": 2.0}.get(length, 1.0)
        )

        full_text = ""
        try:
            for token in self.abstractor.generate_stream(
                system_prompt=persona_obj.system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                delay=delay,
            ):
                full_text += token
                yield _sse("token", {"text": token})
        except Exception as exc:
            # fall back to extractive on LLM failure
            yield _sse("done", {
                "summary": str(result.get("summary", "")),
                "mode": mode,
                "persona": persona,
                "confidence": 1.0,
                "flagged_entities": [],
                "fallback": "extractive",
                "error": str(exc),
            })
            return

        # stage 3: verify (hybrid only)
        confidence = 1.0
        flagged: list[str] = []
        if mode == "hybrid":
            try:
                verification = self.verifier.verify(text, full_text)
                confidence = verification.confidence
                flagged = verification.flagged_entities
            except Exception:
                pass

        yield _sse("done", {
            "summary": full_text,
            "mode": mode,
            "persona": persona,
            "confidence": confidence,
            "flagged_entities": flagged,
        })


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
