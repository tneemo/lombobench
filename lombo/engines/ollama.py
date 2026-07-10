"""Ollama implementation of the common engine contract.

The existing transparent proxy remains unchanged.  This class is for benchmark
code that wants to invoke Ollama through the same contract as local engines.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..capture import StreamAccumulator
from .base import Engine, GenerationResult


class OllamaEngine(Engine):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=None)

    def generate(self, prompt: str | list[dict[str, Any]], **params: Any) -> GenerationResult:
        endpoint = "/api/chat" if isinstance(prompt, list) else "/api/generate"
        body: dict[str, Any] = {"model": self.model, "stream": False, "options": params}
        body["messages" if isinstance(prompt, list) else "prompt"] = prompt
        response = self._client.post(endpoint, json=body)
        response.raise_for_status()
        accumulator = StreamAccumulator(endpoint, stream=False)
        accumulator.feed(response.content)
        return accumulator.result()  # type: ignore[return-value]

    def tokenize(self, text: str) -> list[int]:
        # Ollama's stable public API reports token counts, but does not expose
        # the token ids for an arbitrary input.  Keep the common method honest
        # instead of silently using a different tokenizer than the model.
        raise NotImplementedError("Ollama does not expose input token ids via its public API")

    def get_model_info(self) -> dict[str, Any]:
        response = self._client.post("/api/show", json={"model": self.model})
        response.raise_for_status()
        return {"engine": "ollama", "model": self.model, **response.json()}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaEngine":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
