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

    def tokenizer(text, return_tensors):
        return {"input_ids": FakeTensor([11, 12, 13], (1, 3), "torch.int64")}

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


# --- Multi-layer diagnostics (correlational logit lens) -------------------
# Read-only, without NNsight: uses forward hooks and projects each layer's residual
# through the final norm + lm_head. Tested on a tiny Llama with random weights
# (no downloads); skipped when torch/transformers are not installed.


def _tiny_llama_engine():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=3, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).eval()

    class TinyTok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 5, 9, 3]])}

        def encode(self, text, add_special_tokens=False):
            return [7]  # target token resolves to id 7

        def decode(self, ids):
            return f"<{ids[0]}>"

    engine = TransformersEngine.__new__(TransformersEngine)
    engine.model_name = "tiny/llama"
    engine.model = model
    engine.tokenizer = TinyTok()
    engine._torch = torch
    return engine, model, torch


def test_diagnostics_logit_lens_matches_real_logits():
    engine, model, torch = _tiny_llama_engine()
    assert engine.supports_diagnostics() is True

    result = engine.trace_all_layers("hi", target_token="X", top_k=5)

    assert result["num_layers"] == 3
    assert result["final_norm_applied"] is True
    assert result["target_token"] == "X"
    assert result["target_token_id"] == 7
    assert result["tokens"] == [1, 5, 9, 3]

    layers = result["layers"]
    assert layers[0]["delta_norm"] == 0.0
    for entry in layers:
        assert len(entry["top"]) == 5  # top_k respected
        assert set(entry) == {"index", "top", "hidden_norm", "delta_norm", "entropy",
                              "target_rank", "target_prob"}
        assert entry["entropy"] >= 0.0
        assert set(entry["top"][0]) == {"token_id", "token", "prob"}  # no raw vectors
        assert 1 <= entry["target_rank"] <= 32
        assert 0.0 <= entry["target_prob"] <= 1.0

    # correctness anchor: the final-layer lens reproduces the model's actual argmax
    with torch.inference_mode():
        real = model(input_ids=torch.tensor([[1, 5, 9, 3]])).logits[0, -1]
    assert layers[-1]["top"][0]["token_id"] == int(real.argmax())


def test_diagnostics_without_final_norm(monkeypatch):
    engine, _model, _torch = _tiny_llama_engine()
    monkeypatch.setattr(engine, "_final_norm_path", lambda: None)

    result = engine.trace_all_layers("hi")

    assert result["final_norm_applied"] is False
    assert result["target_token"] is None
    assert result["num_layers"] == 3
    assert result["layers"][0]["target_rank"] is None


def test_diagnostics_requires_supported_layout():
    torch = pytest.importorskip("torch")
    engine = TransformersEngine.__new__(TransformersEngine)
    engine.model = SimpleNamespace(lm_head=object())  # no known ModuleList
    engine._torch = torch
    assert engine.supports_diagnostics() is False
    with pytest.raises(RuntimeError):
        engine.trace_all_layers("hi")


# --- Real activation capture (norm/mean/sparsity per layer×module) ----
# Same tiny Llama with random weights as the diagnostics suite above.


def test_capture_activations_reports_attn_mlp_resid_per_layer():
    engine, _model, _torch = _tiny_llama_engine()
    assert engine.supports_activations() is True

    result = engine.capture_activations("hi", position=-1)

    assert result["num_layers"] == 3
    assert result["modules"] == ["attn", "mlp", "resid"]
    assert result["tokens"] == ["<1>", "<5>", "<9>", "<3>"]
    assert len(result["layers"]) == 3
    for entry in result["layers"]:
        assert set(entry) == {"index", "attn", "mlp", "resid"}
        for module in ("attn", "mlp", "resid"):
            stats = entry[module]
            assert set(stats) == {"l2", "mean", "abs_mean", "max", "sparsity"}
            assert stats["l2"] >= 0.0
            assert 0.0 <= stats["sparsity"] <= 1.0


def test_capture_activations_rejects_out_of_range_position():
    engine, _model, _torch = _tiny_llama_engine()
    with pytest.raises(IndexError):
        engine.capture_activations("hi", position=99)


def test_inspect_activation_returns_top_dims_and_histogram():
    engine, _model, _torch = _tiny_llama_engine()

    result = engine.inspect_activation("hi", layer_index=1, module="mlp",
                                       position=-1, top_k_dims=4, bins=8)

    assert result["layer_index"] == 1
    assert result["module"] == "mlp"
    assert set(result["stats"]) == {"l2", "mean", "abs_mean", "max", "sparsity", "std"}
    assert len(result["top_dims"]) == 4
    for dim in result["top_dims"]:
        assert set(dim) == {"dim", "value", "zscore", "percentile"}
        assert 0.0 <= dim["percentile"] <= 100.0
    assert len(result["histogram"]["counts"]) == 8
    assert len(result["histogram"]["edges"]) == 9
    assert sum(result["histogram"]["counts"]) == 16  # hidden_size from the test config


def test_inspect_activation_rejects_unknown_module():
    engine, _model, _torch = _tiny_llama_engine()
    with pytest.raises(ValueError):
        engine.inspect_activation("hi", layer_index=0, module="bogus")


def test_inspect_activation_rejects_invalid_layer():
    engine, _model, _torch = _tiny_llama_engine()
    with pytest.raises(IndexError):
        engine.inspect_activation("hi", layer_index=99, module="mlp")


def test_activations_requires_supported_layout():
    torch = pytest.importorskip("torch")
    engine = TransformersEngine.__new__(TransformersEngine)
    engine.model = SimpleNamespace(lm_head=object())
    engine._torch = torch
    assert engine.supports_activations() is False
    with pytest.raises(RuntimeError):
        engine.capture_activations("hi")
