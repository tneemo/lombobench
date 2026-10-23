"""Parse requests and accumulate Ollama response streams.

Supports:
- /api/generate and /api/chat (NDJSON when streaming, single JSON otherwise);
- /v1/chat/completions (SSE `data: ...` when streaming, JSON otherwise).

`message.thinking` / `thinking` (reasoning models) is accumulated in a separate
field from the response — never concatenated.
"""

from __future__ import annotations

import json
from typing import Any

from .baseline import OPENAI_TO_OLLAMA

CAPTURED_ENDPOINTS = {"/api/generate", "/api/chat", "/v1/chat/completions"}

# Top-level sampling parameters in the OpenAI-compatible endpoint body.
_OPENAI_TOP_LEVEL_PARAMS = set(OPENAI_TO_OLLAMA) | {"n", "logit_bias", "top_logprobs"}


def parse_request(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized request fields (the raw body is stored separately)."""
    info: dict[str, Any] = {
        "model": body.get("model"),
        "stream": bool(body.get("stream", endpoint != "/v1/chat/completions")),
        "requested_params": {},
        "requested_params_ollama": {},  # names mapped for the inferred layer
        "system_prompt": None,
        "prompt_or_messages": None,
    }
    if endpoint == "/v1/chat/completions":
        info["stream"] = bool(body.get("stream", False))
        requested = {k: v for k, v in body.items() if k in _OPENAI_TOP_LEVEL_PARAMS}
        info["requested_params"] = requested
        info["requested_params_ollama"] = {
            OPENAI_TO_OLLAMA[k]: v for k, v in requested.items() if k in OPENAI_TO_OLLAMA
        }
        messages = body.get("messages", [])
        info["prompt_or_messages"] = messages
        info["system_prompt"] = _system_from_messages(messages)
    else:
        options = body.get("options") or {}
        info["requested_params"] = dict(options)
        info["requested_params_ollama"] = dict(options)
        if endpoint == "/api/chat":
            messages = body.get("messages", [])
            info["prompt_or_messages"] = messages
            info["system_prompt"] = _system_from_messages(messages)
        else:  # /api/generate
            info["prompt_or_messages"] = body.get("prompt")
            info["system_prompt"] = body.get("system")
    return info


def _system_from_messages(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    parts = [m.get("content") for m in messages
             if isinstance(m, dict) and m.get("role") == "system" and m.get("content")]
    return "\n".join(parts) if parts else None


class StreamAccumulator:
    """Accumulate response bytes and extract text/thinking/metrics at the end."""

    def __init__(self, endpoint: str, stream: bool) -> None:
        self.endpoint = endpoint
        self.stream = stream
        self._raw = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._raw.extend(chunk)

    def result(self) -> dict[str, Any]:
        """Return {response_text, thinking_text, final_chunk, metrics, done_reason, parse_error}."""
        raw = bytes(self._raw)
        try:
            if self.endpoint == "/v1/chat/completions":
                parsed = self._parse_openai(raw) if self.stream else self._parse_openai_single(raw)
            else:
                parsed = self._parse_ndjson(raw) if self.stream else self._parse_single(raw)
        except Exception as exc:
            return {"response_text": None, "thinking_text": None, "final_chunk": None,
                    "metrics": {}, "done_reason": None,
                    "parse_error": f"{type(exc).__name__}: {exc}"}
        parsed.setdefault("parse_error", None)
        return parsed

    # ---- Native Ollama ----

    def _parse_ndjson(self, raw: bytes) -> dict[str, Any]:
        response, thinking, final = [], [], None
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            self._collect_native(obj, response, thinking)
            if obj.get("done"):
                final = obj
        return self._finish_native(response, thinking, final)

    def _parse_single(self, raw: bytes) -> dict[str, Any]:
        obj = json.loads(raw)
        response, thinking = [], []
        self._collect_native(obj, response, thinking)
        return self._finish_native(response, thinking, obj)

    @staticmethod
    def _collect_native(obj: dict, response: list, thinking: list) -> None:
        msg = obj.get("message")
        if isinstance(msg, dict):
            if msg.get("content"):
                response.append(msg["content"])
            if msg.get("thinking"):
                thinking.append(msg["thinking"])
        if obj.get("response"):
            response.append(obj["response"])
        if obj.get("thinking"):
            thinking.append(obj["thinking"])

    @staticmethod
    def _finish_native(response: list, thinking: list, final: dict | None) -> dict[str, Any]:
        metrics = {}
        done_reason = None
        if final:
            for key in ("total_duration", "load_duration", "prompt_eval_count",
                        "prompt_eval_duration", "eval_count", "eval_duration"):
                if key in final:
                    metrics[key] = final[key]
            done_reason = final.get("done_reason")
        return {
            "response_text": "".join(response) or None,
            "thinking_text": "".join(thinking) or None,
            "final_chunk": final,
            "metrics": metrics,
            "done_reason": done_reason,
        }

    # ---- OpenAI-compat ----

    def _parse_openai(self, raw: bytes) -> dict[str, Any]:
        response, thinking, final = [], [], None
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                continue
            obj = json.loads(payload)
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    response.append(delta["content"])
                if delta.get("reasoning") or delta.get("thinking"):
                    thinking.append(delta.get("reasoning") or delta.get("thinking"))
                if choice.get("finish_reason"):
                    final = obj
            if obj.get("usage"):
                final = obj
        return self._finish_openai(response, thinking, final)

    def _parse_openai_single(self, raw: bytes) -> dict[str, Any]:
        obj = json.loads(raw)
        response, thinking = [], []
        for choice in obj.get("choices", []):
            msg = choice.get("message") or {}
            if msg.get("content"):
                response.append(msg["content"])
            if msg.get("reasoning") or msg.get("thinking"):
                thinking.append(msg.get("reasoning") or msg.get("thinking"))
        return self._finish_openai(response, thinking, obj)

    @staticmethod
    def _finish_openai(response: list, thinking: list, final: dict | None) -> dict[str, Any]:
        metrics = {}
        done_reason = None
        if final:
            usage = final.get("usage") or {}
            if "prompt_tokens" in usage:
                metrics["prompt_eval_count"] = usage["prompt_tokens"]
            if "completion_tokens" in usage:
                metrics["eval_count"] = usage["completion_tokens"]
            for choice in final.get("choices", []):
                if choice.get("finish_reason"):
                    done_reason = choice["finish_reason"]
        return {
            "response_text": "".join(response) or None,
            "thinking_text": "".join(thinking) or None,
            "final_chunk": final,
            "metrics": metrics,
            "done_reason": done_reason,
        }


def tokens_per_second(metrics: dict[str, Any]) -> float | None:
    count, duration = metrics.get("eval_count"), metrics.get("eval_duration")
    if count and duration:
        return count / (duration / 1e9)
    return None
