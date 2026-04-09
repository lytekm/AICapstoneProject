"""Three-stage summarization pipeline: extract -> abstract -> verify.

This is the core orchestration module. It wires together Kevin's
TextRank+MMR extractor, the LLM-based abstractor, and the NER
verifier into a single call. Each stage is optional depending on
the mode:
  - extractive: just TextRank+MMR, no LLM needed
  - abstractive: extract first (to get key sentences), then LLM rewrites
  - hybrid: extract -> LLM rewrite -> NER consistency check
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from src.abstractor import AbstractorBase, create_abstractor
from src.personas import LENGTH_MULTIPLIERS, format_prompt, get_persona
from src.summarizer_model import TextRankMMRSummarizer
from src.verifier import NERVerifier

VALID_MODES = {"extractive", "abstractive", "hybrid"}
VALID_LENGTHS = {"brief", "standard", "detailed"}


@dataclass
class PipelineResult:
    """Everything the API layer needs to build a response."""

    summary: str
    mode: str
    persona: str
    # None means the summary was not verified; otherwise the verifier score.
    confidence: float | None = None
    flagged_entities: list[str] = field(default_factory=list)
    # the raw extracted sentences before LLM rewriting
    extractive_sentences: list[str] = field(default_factory=list)
    # catch-all for error info, fallback flags, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


class SummarizationPipeline:
    """Three-stage summarization pipeline: extract, abstract, verify."""

    def __init__(
        self,
        summarizer: TextRankMMRSummarizer | None = None,
        abstractor: AbstractorBase | None = None,
        verifier: NERVerifier | None = None,
    ) -> None:
        # each component can be injected for testing, otherwise use defaults
        self.summarizer = summarizer or TextRankMMRSummarizer()
        self.abstractor = abstractor or create_abstractor()
        self.verifier = verifier or NERVerifier()

    def _extract(self, text: str, k: int) -> dict[str, Any]:
        """Run TextRank+MMR to pick the top-k sentences."""
        return self.summarizer.summarize(text, k=k)

    def _abstract(
        self,
        sentences: list[str],
        persona_name: str,
        length: str,
    ) -> str:
        """Send extracted sentences to the LLM for a persona-styled rewrite."""
        persona = get_persona(persona_name)
        user_prompt = format_prompt(persona, sentences, length)
        # scale the token budget based on requested length
        max_tokens = int(
            persona.max_tokens_hint
            * LENGTH_MULTIPLIERS.get(length, 1.0)
        )
        return self.abstractor.generate(
            system_prompt=persona.system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

    def _effective_extractive_k(self, k: int, length: str) -> int:
        """Scale the extractive sentence budget using the selected length."""
        multiplier = LENGTH_MULTIPLIERS.get(length, 1.0)
        return max(1, min(20, ceil(k * multiplier)))

    def _verify(self, source_text: str, summary_text: str) -> tuple[float | None, list[str]]:
        """Run NER verification when available and return a normalized result."""
        if not self.verifier.available:
            return None, []

        try:
            verification = self.verifier.verify(source_text, summary_text)
        except Exception:
            return None, []

        return verification.confidence, verification.flagged_entities

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

        # validate persona early so we fail fast before doing expensive work
        get_persona(persona)

        if mode == "extractive":
            return self._run_extractive(text, persona, length, k)
        elif mode == "abstractive":
            return self._run_abstractive(text, persona, length, k)
        else:
            return self._run_hybrid(text, persona, length, k)

    def _run_extractive(
        self, text: str, persona: str, length: str, k: int
    ) -> PipelineResult:
        """Pure extractive: just return the top-k sentences as-is."""
        effective_k = self._effective_extractive_k(k, length)
        result = self._extract(text, effective_k)
        selected = result.get("selected_indices", [])
        sentences = result.get("sentences", [])
        # bounds-check in case the article had fewer sentences than k
        extracted = [sentences[i] for i in selected if i < len(sentences)]

        return PipelineResult(
            summary=str(result.get("summary", "")),
            mode="extractive",
            persona=persona,
            extractive_sentences=extracted,
            metadata={"requested_k": k, "effective_k": effective_k, "length": length},
        )

    def _run_abstractive(
        self, text: str, persona: str, length: str, k: int
    ) -> PipelineResult:
        """Extract key sentences, then let the LLM rewrite them.

        Even abstractive mode starts with extraction -- we don't send the
        entire article to the LLM, just the most relevant sentences.
        This keeps the prompt short and focused.
        """
        result = self._extract(text, k)
        selected = result.get("selected_indices", [])
        sentences = result.get("sentences", [])
        extracted = [sentences[i] for i in selected if i < len(sentences)]

        try:
            summary = self._abstract(extracted, persona, length)
        except Exception as exc:
            # if the LLM is down, fall back to extractive rather than crashing
            return PipelineResult(
                summary=str(result.get("summary", "")),
                mode="abstractive",
                persona=persona,
                extractive_sentences=extracted,
                metadata={"abstractor_error": str(exc), "fallback": "extractive"},
            )

        confidence, flagged = self._verify(text, summary)
        return PipelineResult(
            summary=summary,
            mode="abstractive",
            persona=persona,
            confidence=confidence,
            flagged_entities=flagged,
            extractive_sentences=extracted,
        )

    def _run_hybrid(
        self, text: str, persona: str, length: str, k: int
    ) -> PipelineResult:
        """All three stages: extract -> abstract -> verify.

        This is the full pipeline. After the LLM rewrites, we run NER
        to check for hallucinated entities (names, orgs, etc. that
        appear in the summary but not the source).
        """
        # Stage 1: pick the most relevant sentences
        result = self._extract(text, k)
        selected = result.get("selected_indices", [])
        sentences = result.get("sentences", [])
        extracted = [sentences[i] for i in selected if i < len(sentences)]

        # Stage 2: LLM rewrite with persona styling
        try:
            summary = self._abstract(extracted, persona, length)
        except Exception as exc:
            # graceful degradation: extractive output is still useful
            return PipelineResult(
                summary=str(result.get("summary", "")),
                mode="hybrid",
                persona=persona,
                extractive_sentences=extracted,
                metadata={"abstractor_error": str(exc), "fallback": "extractive"},
            )

        # Stage 3: NER-based hallucination check
        confidence, flagged = self._verify(text, summary)

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
        """Yield SSE-formatted events for streaming responses.

        The frontend listens for three event types:
          - "meta": sent first, contains mode/persona info
          - "token": one chunk of the LLM output (for progressive rendering)
          - "done": final event with full summary, confidence, flagged entities
        Extractive mode skips straight to "done" since there's no streaming.
        """
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

        # extraction is always the first step, even for streaming
        effective_k = self._effective_extractive_k(k, length) if mode == "extractive" else k
        result = self._extract(text, effective_k)
        selected = result.get("selected_indices", [])
        sentences = result.get("sentences", [])
        extracted = [sentences[i] for i in selected if i < len(sentences)]

        # extractive mode has nothing to stream -- just send the result
        if mode == "extractive":
            yield _sse("done", {
                "summary": str(result.get("summary", "")),
                "mode": "extractive",
                "persona": persona,
                "confidence": None,
                "flagged_entities": [],
                "effective_k": effective_k,
            })
            return

        # tell the frontend what's coming before tokens start flowing
        yield _sse("meta", {"mode": mode, "persona": persona})

        # build the prompt the same way _abstract() does
        persona_obj = get_persona(persona)
        user_prompt = format_prompt(persona_obj, extracted, length)
        max_tokens = int(
            persona_obj.max_tokens_hint
            * LENGTH_MULTIPLIERS.get(length, 1.0)
        )

        # accumulate the full text so we can verify it after streaming
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
            # LLM died mid-stream; send extractive fallback
            yield _sse("done", {
                "summary": str(result.get("summary", "")),
                "mode": mode,
                "persona": persona,
                "confidence": None,
                "flagged_entities": [],
                "fallback": "extractive",
                "error": str(exc),
            })
            return

        # run NER verification on the complete streamed output when supported
        confidence: float | None = None
        flagged: list[str] = []
        if mode in {"abstractive", "hybrid"}:
            confidence, flagged = self._verify(text, full_text)

        yield _sse("done", {
            "summary": full_text,
            "mode": mode,
            "persona": persona,
            "confidence": confidence,
            "flagged_entities": flagged,
        })


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format a single Server-Sent Event.

    SSE spec requires "event: <type>\\ndata: <json>\\n\\n".
    The double newline at the end signals the end of one event.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
