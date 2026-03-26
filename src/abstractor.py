"""LLM abstraction layer for summary rewriting.

Two implementations:
  - MockAbstractor: deterministic, no network calls, used in tests and CI
  - Abstractor: real OpenAI-compatible client for vLLM/Ollama endpoints

The factory function create_abstractor() picks the right one based on
the USE_MOCK_LLM env var. This way the rest of the codebase never
imports a specific class -- it just calls the factory.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class AbstractorConfig:
    """Connection settings for the real LLM endpoint."""

    base_url: str = ""
    model: str = "default"
    # low temperature for more focused, less creative summaries
    temperature: float = 0.3
    max_tokens: int = 512

    def __post_init__(self) -> None:
        # fall back to env var so we don't hardcode the vLLM address
        if not self.base_url:
            self.base_url = os.environ.get(
                "VLLM_BASE_URL", "http://localhost:8000/v1"
            )


class AbstractorBase(ABC):
    """Base class for LLM-based abstractive summarizers."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> str:
        """Generate an abstractive summary from the given prompts."""

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        delay: float = 0.02,
    ) -> Iterator[str]:
        """Yield summary tokens one at a time.

        Default implementation: call generate() then split on spaces.
        The real Abstractor overrides this with actual streaming.
        The delay param is for demos/presentations so tokens appear gradually.
        """
        text = self.generate(system_prompt, user_prompt, max_tokens)
        for word in text.split():
            if delay > 0:
                time.sleep(delay)
            # add trailing space so words don't smash together in the frontend
            yield word + " "


class MockAbstractor(AbstractorBase):
    """Deterministic mock for testing and offline development.

    Strips bullet-point markers from the prompt, grabs up to 3 sentences,
    and prefixes with "[Mock Summary]". No randomness, no network calls.
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> str:
        lines = user_prompt.strip().splitlines()
        sentences: list[str] = []
        for line in lines:
            # the prompt formats sentences as "- sentence here"
            cleaned = re.sub(r"^-\s*", "", line.strip())
            # skip the instruction line itself
            if cleaned and not cleaned.startswith("Summarize"):
                sentences.append(cleaned)

        # cap at 3 so mock output stays short and predictable
        picked = sentences[:3] if sentences else ["No content provided."]
        return "[Mock Summary] " + " ".join(picked)


class Abstractor(AbstractorBase):
    """Real LLM client using the OpenAI-compatible API (vLLM, Ollama, etc).

    Uses the openai Python SDK because vLLM exposes an OpenAI-compatible
    endpoint. This means we can swap between vLLM, Ollama, or actual
    OpenAI without changing any code -- just point VLLM_BASE_URL at it.
    """

    def __init__(self, config: AbstractorConfig | None = None) -> None:
        self.config = config or AbstractorConfig()
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            self.client = OpenAI(
                base_url=self.config.base_url,
                # vLLM doesn't need a real API key, but the SDK requires one
                api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            )
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for the real Abstractor. "
                "Install it with: pip install openai"
            ) from exc

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return (choice.message.content or "").strip()

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        delay: float = 0.0,
    ) -> Iterator[str]:
        """Stream tokens from the real LLM via the OpenAI SDK.

        Each chunk from the API contains a delta with (usually) one token.
        We yield them immediately so the frontend can render progressively.
        """
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


def create_abstractor(use_mock: bool | None = None) -> AbstractorBase:
    """Factory for creating the appropriate abstractor.

    Priority: explicit param > USE_MOCK_LLM env var > default (mock).
    Default is mock so tests and CI work without a running LLM.
    Set USE_MOCK_LLM=0 to hit the real endpoint.
    """
    if use_mock is not None:
        return MockAbstractor() if use_mock else Abstractor()

    env_val = os.environ.get("USE_MOCK_LLM", "1").lower()
    if env_val in ("0", "false", "no"):
        return Abstractor()

    return MockAbstractor()
