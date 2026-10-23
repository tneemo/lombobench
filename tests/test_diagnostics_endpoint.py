"""Endpoint contract for the multi-layer diagnostics (logit lens).

Exercises the guards and payload passthrough of ``POST /requests/{id}/diagnostics``
without loading a model — the DB connection and engine are faked, mirroring the
style of ``test_proxy_engines``.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from lombo.analysis import DiagnosticsRequest, diagnostics_request


class FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConn:
    """Dispatches the two SELECTs the endpoint runs, by SQL fragment."""
    def __init__(self, engine_name="transformers", prompt=None):
        self.engine_name = engine_name
        self.prompt = prompt if prompt is not None else [{"role": "user", "content": "hi"}]

    def execute(self, sql, params):
        if "FROM requests" in sql:
            return FakeCursor({
                "model": "fake/model",
                "lab_session_id": "sess-1",
                "prompt_or_messages_json": json.dumps(self.prompt),
            })
        if "FROM lab_sessions" in sql:
            return FakeCursor({"base_config_json": json.dumps({"engine": self.engine_name})})
        return FakeCursor(None)


class FakeDiagEngine:
    def __init__(self, supports=True):
        self._supports = supports
        self.calls = []

    def supports_diagnostics(self):
        return self._supports

    def trace_all_layers(self, prompt, target_token=None, top_k=10):
        self.calls.append((prompt, target_token, top_k))
        return {"num_layers": 1, "target_token": target_token, "target_token_id": None,
                "final_norm_applied": True, "tokens": [1], "layers": [
                    {"index": 0, "top": [{"token_id": 1, "token": "a", "prob": 1.0}],
                     "hidden_norm": 1.0, "delta_norm": 0.0,
                     "target_rank": None, "target_prob": None}],
                "model": {"engine": "transformers"}}


def _request(state):
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _state(conn, engines):
    return SimpleNamespace(conn=conn, transformer_engines=engines)


def test_diagnostics_endpoint_returns_layer_payload():
    engine = FakeDiagEngine()
    state = _state(FakeConn(), {"fake/model": engine})
    body = DiagnosticsRequest(target_token="Paris", top_k=5)

    result = asyncio.run(diagnostics_request(_request(state), "req-1", body))

    assert result["num_layers"] == 1
    assert result["target_token"] == "Paris"
    assert engine.calls == [([{"role": "user", "content": "hi"}], "Paris", 5)]


def test_diagnostics_endpoint_rejects_non_transformers():
    state = _state(FakeConn(engine_name="ollama"), {})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(diagnostics_request(_request(state), "req-1", DiagnosticsRequest()))
    assert exc.value.status_code == 409


def test_diagnostics_endpoint_requires_loaded_engine():
    state = _state(FakeConn(), {})  # engine not loaded
    with pytest.raises(HTTPException) as exc:
        asyncio.run(diagnostics_request(_request(state), "req-1", DiagnosticsRequest()))
    assert exc.value.status_code == 409


def test_diagnostics_endpoint_422_when_layout_unsupported():
    state = _state(FakeConn(), {"fake/model": FakeDiagEngine(supports=False)})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(diagnostics_request(_request(state), "req-1", DiagnosticsRequest()))
    assert exc.value.status_code == 422
