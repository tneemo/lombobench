"""Tests for routing local engines through the proxy response contract."""

import asyncio
import json
from types import SimpleNamespace

from lombo.proxy import _response_chunks, _transformers_upstream


class FakeTransformersEngine:
    def generate(self, prompt, **options):
        assert prompt[-1] == {"role": "user", "content": "hi"}
        assert options == {"temperature": 0.2}
        return {
            "response_text": "hello",
            "thinking_text": None,
            "final_chunk": {"done": True, "done_reason": "stop",
                            "eval_count": 2, "eval_duration": 1_000_000_000},
            "metrics": {}, "done_reason": "stop", "parse_error": None,
        }


class FailingTransformersEngine:
    def generate(self, prompt, **options):
        return {"response_text": None, "thinking_text": None,
                "final_chunk": None, "metrics": {}, "done_reason": None,
                "parse_error": "RuntimeError: generation failed"}


def test_transformers_response_uses_ollama_chat_shape():
    app = SimpleNamespace(state=SimpleNamespace(
        transformer_engines={"fake/model": FakeTransformersEngine()},
        transformer_engine_lock=asyncio.Lock(),
    ))
    body = json.dumps({
        "model": "fake/model",
        "messages": [{"role": "user", "content": "hi"}],
        "options": {"temperature": 0.2},
        "stream": True,
    }).encode()

    response = asyncio.run(_transformers_upstream(app, "/api/chat", body))
    payload = response.json()

    assert response.status_code == 200
    assert payload["model"] == "fake/model"
    assert payload["message"]["content"] == "hello"
    assert payload["eval_count"] == 2

    async def collect():
        return b"".join([chunk async for chunk in
                         _response_chunks(response, buffered=True)])

    assert json.loads(asyncio.run(collect()))["message"]["content"] == "hello"


def test_transformers_generation_error_is_not_a_silent_200():
    app = SimpleNamespace(state=SimpleNamespace(
        transformer_engines={"fake/model": FailingTransformersEngine()},
        transformer_engine_status={}, transformer_engine_lock=asyncio.Lock(),
    ))
    body = json.dumps({"model": "fake/model", "messages": [], "stream": True}).encode()

    response = asyncio.run(_transformers_upstream(app, "/api/chat", body))

    assert response.status_code == 500
    assert response.json()["error"] == "RuntimeError: generation failed"
    assert app.state.transformer_engine_status["fake/model"]["state"] == "error"
