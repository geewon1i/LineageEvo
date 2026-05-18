"""Provider-neutral LLM client contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    content: str


class LLMClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        ...


class MockLLMClient:
    """Queue-backed mock client used by tests and smoke runs."""

    def __init__(self, responses: list[LLMResponse | str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if not self.responses:
            return LLMResponse("{}")
        response = self.responses.pop(0)
        if isinstance(response, str):
            return LLMResponse(response)
        return response
