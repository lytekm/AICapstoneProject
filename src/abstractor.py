from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class AbstractorConfig:
    base_url: str = ""
    model: str = "default"
    temperature: float = 0.3
    max_tokens: int = 512

    def __post_init__(self) -> None:
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
        """Yield summary tokens one at a time. Default impl splits generate() output."""
        text = self.generate(system_prompt, user_prompt, max_tokens)
        for word in text.split():
            if delay > 0:
                time.sleep(delay)
            yield word + " "


class MockAbstractor(AbstractorBase):
    """Deterministic mock for testing and offline development."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
    ) -> str:
        lines = user_prompt.strip().splitlines()
        sentences: list[str] = []
        for line in lines:
            cleaned = re.sub(r"^-\s*", "", line.strip())
            if cleaned and not cleaned.startswith("Summarize"):
                sentences.append(cleaned)

        picked = sentences[:3] if sentences else ["No content provided."]
        return "[Mock Summary] " + " ".join(picked)


class Abstractor(AbstractorBase):
    """Real LLM client using the OpenAI-compatible API (vLLM, Ollama, etc)."""

    def __init__(self, config: AbstractorConfig | None = None) -> None:
        self.config = config or AbstractorConfig()
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            self.client = OpenAI(
                base_url=self.config.base_url,
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
        """Stream tokens from the real LLM via the OpenAI SDK."""
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
    """
    if use_mock is not None:
        return MockAbstractor() if use_mock else Abstractor()

    env_val = os.environ.get("USE_MOCK_LLM", "1").lower()
    if env_val in ("0", "false", "no"):
        return Abstractor()

    return MockAbstractor()
