"""Thin OpenAI client wrapper.

Reads configuration from environment variables loaded via python-dotenv from a
local `.env` file (see `.env.example`). The `openai` package is imported lazily
so that pure evaluation functions stay importable even when the SDK is missing
or the API key is not configured.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2.0


class LLMClientError(RuntimeError):
    """Raised when the model backend cannot be reached."""


@dataclass
class Completion:
    """Structured result of a single model call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Minimal OpenAI chat-completions wrapper with cost-relevant usage info."""

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMClientError(
                "OPENAI_API_KEY is not set. Create a `.env` file from `.env.example` "
                "and add your key."
            )
        self.default_model = default_model or os.getenv("EVAL_MODEL", "gpt-4o-mini")
        self._client = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # lazy import

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def complete(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> Completion:
        """Call the chat model and return a structured Completion.

        A small exponential-backoff retry handles transient 429/5xx responses.
        """
        model = model or self.default_model
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**kwargs)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                usage = response.usage
                return Completion(
                    text=response.choices[0].message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    elapsed_ms=elapsed_ms,
                )
            except Exception as exc:  # noqa: BLE001 - surface as client error below
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BASE_BACKOFF_SECONDS * (2 ** attempt))

        raise LLMClientError(f"Model call failed for model '{model}': {last_error}") from last_error