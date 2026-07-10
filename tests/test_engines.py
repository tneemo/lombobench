"""Backend contract tests that do not require model downloads."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from lombo.engines import Engine, TransformersEngine


class StubEngine(Engine):
    def generate(self, prompt, **params):
        return {"response_text": str(prompt), "thinking_text": None,
                "final_chunk": None, "metrics": {}, "done_reason": "stop",
                "parse_error": None}

    def tokenize(self, text):
        return [len(text)]

    def get_model_info(self):
        return {"engine": "stub"}


def test_engine_has_safe_trace_default():
    engine = StubEngine()
    assert engine.supports_trace() is False
    with pytest.raises(NotImplementedError):
        engine.trace("prompt")


def test_transformers_maps_shared_generation_parameters():
    mapped = TransformersEngine._generation_params({
        "seed": 7, "temperature": 0, "top_p": 0.8, "top_k": 20,
        "num_predict": 42, "repeat_penalty": 1.1, "unsupported": "ignored",
    })
    assert mapped == {"top_p": 0.8, "top_k": 20, "max_new_tokens": 42,
                      "repetition_penalty": 1.1, "do_sample": False}


class FakeTensor:
    def __init__(self, values, shape, dtype="torch.float32"):
        self.values = values
        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, key):
        return self

    def tolist(self):
        return self.values

    def detach(self):
        return self

    def cpu(self):
        return self


class SavedOutput:
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, key):
        return self

    def save(self):
        return self.tensor


class FakeLanguageModel:
    last_trace_input = None

    def __init__(self, model, tokenizer):
        activation = FakeTensor([], (1, 3, 4), "torch.bfloat16")
        logits = FakeTensor([0.1, 0.2, 0.3], (3,), "torch.float32")
        self.model = SimpleNamespace(layers=[
            SimpleNamespace(output=SavedOutput(activation)),
            SimpleNamespace(output=SavedOutput(activation)),
        ])
        self.lm_head = SimpleNamespace(output=SavedOutput(logits))

    def trace(self, encoded):
        FakeLanguageModel.last_trace_input = encoded
        return FakeTraceContext()


class FakeTraceContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _trace_engine(monkeypatch):
    nnsight = ModuleType("nnsight")
    nnsight.LanguageModel = FakeLanguageModel
    monkeypatch.setitem(sys.modules, "nnsight", nnsight)

    parameter = SimpleNamespace(device="cpu", dtype="torch.bfloat16")
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[object(), object()]),
        lm_head=object(),
        parameters=lambda: iter([parameter]),
    )
    tokenizer = lambda text, return_tensors: {
        "input_ids": FakeTensor([11, 12, 13], (1, 3), "torch.int64")
    }
    engine = TransformersEngine.__new__(TransformersEngine)
    engine.model_name = "fake/model"
    engine.model = model
    engine.tokenizer = tokenizer
    return engine


def test_transformers_trace_is_opt_in_and_serializable(monkeypatch):
    engine = _trace_engine(monkeypatch)
    assert engine.supports_trace() is True

    result = engine.trace("hello", layer_index=-1)

    assert result == {
        "layer_index": 1,
        "activation_shape": [1, 3, 4],
        "activation_dtype": "torch.bfloat16",
        "tokens": [11, 12, 13],
        "last_token_logits": [0.1, 0.2, 0.3],
        "model": {"engine": "transformers", "model": "fake/model",
                  "architecture": "SimpleNamespace", "device": "cpu",
                  "dtype": "torch.bfloat16"},
    }


def test_transformers_trace_rejects_invalid_layer(monkeypatch):
    engine = _trace_engine(monkeypatch)
    with pytest.raises(IndexError):
        engine.trace("hello", layer_index=2)


def test_transformers_trace_rejects_unknown_model_layout(monkeypatch):
    engine = _trace_engine(monkeypatch)
    engine.model = SimpleNamespace(lm_head=object())
    assert engine.supports_trace() is False
