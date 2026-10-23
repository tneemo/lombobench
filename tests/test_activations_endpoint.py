"""Endpoint contract for real activation capture.

Exercises the guards and payload passthrough of ``POST /requests/{id}/activations``
and ``.../activations/inspect`` without loading a model — the DB connection and
engine are faked, mirroring the style of ``test_diagnostics_endpoint.py``.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from lombo.analysis import (
    ActivationsInspectRequest,
    ActivationsRequest,
    activations_inspect_request,
    activations_request,
)


class FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConn:
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


class FakeActivationsEngine:
    def __init__(self, supports=True):
        self._supports = supports
        self.calls = []
        self.inspect_calls = []

    def supports_activations(self):
        return self._supports

    def capture_activations(self, prompt, position=-1):
        self.calls.append((prompt, position))
        return {"num_layers": 1, "position": position, "tokens": ["a"],
                "modules": ["attn", "mlp", "resid"],
                "layers": [{"index": 0, "attn": {"l2": 1.0}, "mlp": {"l2": 1.0},
                           "resid": {"l2": 1.0}}],
                "model": {"engine": "transformers"}}

    def inspect_activation(self, prompt, layer_index, module, position=-1,
                           top_k_dims=16, bins=32):
        self.inspect_calls.append((prompt, layer_index, module, position, top_k_dims, bins))
        return {"layer_index": layer_index, "module": module, "position": position,
                "stats": {"l2": 1.0, "mean": 0.0, "abs_mean": 0.1, "max": 2.0,
                         "sparsity": 0.5, "std": 0.3},
                "top_dims": [{"dim": 0, "value": 2.0, "zscore": 3.0, "percentile": 99.0}],
                "histogram": {"edges": [0, 1], "counts": [1], "unit": "zscore"}}


def _request(state):
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _state(conn, engines):
    return SimpleNamespace(conn=conn, transformer_engines=engines)


def test_activations_endpoint_returns_layer_payload():
    engine = FakeActivationsEngine()
    state = _state(FakeConn(), {"fake/model": engine})

    result = asyncio.run(activations_request(_request(state), "req-1", ActivationsRequest(position=-1)))

    assert result["num_layers"] == 1
    assert result["modules"] == ["attn", "mlp", "resid"]
    assert engine.calls == [([{"role": "user", "content": "hi"}], -1)]


def test_activations_endpoint_rejects_non_transformers():
    state = _state(FakeConn(engine_name="ollama"), {})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(activations_request(_request(state), "req-1", ActivationsRequest()))
    assert exc.value.status_code == 409


def test_activations_endpoint_requires_loaded_engine():
    state = _state(FakeConn(), {})  # engine not loaded
    with pytest.raises(HTTPException) as exc:
        asyncio.run(activations_request(_request(state), "req-1", ActivationsRequest()))
    assert exc.value.status_code == 409


def test_activations_endpoint_422_when_layout_unsupported():
    state = _state(FakeConn(), {"fake/model": FakeActivationsEngine(supports=False)})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(activations_request(_request(state), "req-1", ActivationsRequest()))
    assert exc.value.status_code == 422


def test_activations_inspect_returns_detail_payload():
    engine = FakeActivationsEngine()
    state = _state(FakeConn(), {"fake/model": engine})
    body = ActivationsInspectRequest(layer_index=3, module="mlp", position=-1,
                                     top_k_dims=8, bins=16)

    result = asyncio.run(activations_inspect_request(_request(state), "req-1", body))

    assert result["layer_index"] == 3
    assert result["module"] == "mlp"
    assert engine.inspect_calls == [([{"role": "user", "content": "hi"}], 3, "mlp", -1, 8, 16)]


def test_activations_inspect_rejects_invalid_module():
    engine = FakeActivationsEngine()
    state = _state(FakeConn(), {"fake/model": engine})
    body = ActivationsInspectRequest(layer_index=0, module="bogus")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(activations_inspect_request(_request(state), "req-1", body))
    assert exc.value.status_code == 422


def test_activations_inspect_rejects_non_transformers():
    state = _state(FakeConn(engine_name="ollama"), {})
    body = ActivationsInspectRequest(layer_index=0, module="mlp")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(activations_inspect_request(_request(state), "req-1", body))
    assert exc.value.status_code == 409
