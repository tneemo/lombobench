"""Small, backend-neutral inference contract used by benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class GenerationResult(TypedDict):
    """Normalized result consumed by the benchmark/capture layers."""

    response_text: str | None
    thinking_text: str | None
    final_chunk: dict[str, Any] | None
    metrics: dict[str, Any]
    done_reason: str | None
    parse_error: str | None


class Engine(ABC):
    """Minimum contract for an inference backend.

    Implementations own model loading and backend-specific parameter mapping.
    Benchmark code should only depend on this class and ``GenerationResult``.
    """

    @abstractmethod
    def generate(self, prompt: str | list[dict[str, Any]], **params: Any) -> GenerationResult:
        """Generate text from a prompt or chat message list."""

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Return the model token ids for ``text``."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return backend and model metadata."""

    def supports_trace(self) -> bool:
        """Whether ``trace`` can instrument an inference execution."""
        return False

    def trace(self, *args: Any, **kwargs: Any) -> Any:
        """Future NNsight integration point; unsupported by default."""
        raise NotImplementedError(f"{type(self).__name__} does not support tracing")
