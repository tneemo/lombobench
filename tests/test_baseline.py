"""Camadas de parâmetros e parsing do bloco `parameters` — funções puras."""

from lombo.baseline import (
    OLLAMA_BUILTIN_DEFAULTS,
    confirmed_from_response,
    layer_params,
    parse_parameters_raw,
)


def test_parse_parameters_raw_types_and_repeats():
    raw = 'temperature 0.6\nnum_ctx 8192\nstop "<|eot|>"\nstop "<|end|>"\n'
    parsed = parse_parameters_raw(raw)
    assert parsed["temperature"] == 0.6
    assert parsed["num_ctx"] == 8192
    assert parsed["stop"] == ["<|eot|>", "<|end|>"]


def test_parse_parameters_raw_empty():
    assert parse_parameters_raw(None) == {}
    assert parse_parameters_raw("") == {}


def test_layer_params_request_wins_and_diff():
    requested = {"temperature": 0.1, "top_k": 5}
    modelfile = {"temperature": 0.6, "num_ctx": 8192}
    inferred, diff, base = layer_params(requested, modelfile)

    assert inferred["temperature"] == {"value": 0.1, "source": "request", "inferred": True}
    assert inferred["top_k"] == {"value": 5, "source": "request", "inferred": True}
    # sem request nem modelfile → default do Ollama
    assert inferred["top_p"] == {"value": 0.9, "source": "ollama_default", "inferred": True}
    # modelfile vence o default
    assert inferred["num_ctx"] == {"value": 8192, "source": "modelfile", "inferred": True}

    assert diff["temperature"] == {"value": 0.1, "base_value": 0.6, "base_source": "modelfile"}
    assert diff["top_k"] == {"value": 5, "base_value": 40, "base_source": "ollama_default"}
    assert "num_ctx" not in diff  # não foi solicitado, logo não difere do base
    assert base["temperature"] == 0.6


def test_layer_params_no_request_has_empty_diff():
    inferred, diff, _ = layer_params({}, {"temperature": 0.6})
    assert diff == {}
    assert inferred["temperature"]["source"] == "modelfile"
    assert all(v["inferred"] is True for v in inferred.values())


def test_layer_params_request_equal_to_base_not_in_diff():
    inferred, diff, _ = layer_params(
        {"temperature": OLLAMA_BUILTIN_DEFAULTS["temperature"]}, {})
    assert inferred["temperature"]["source"] == "request"
    assert diff == {}


def test_confirmed_only_from_response():
    assert confirmed_from_response(None) == {}
    confirmed = confirmed_from_response(
        {"model": "llama3.2:3b", "done": True, "done_reason": "stop", "eval_count": 42})
    assert confirmed["model"]["value"] == "llama3.2:3b"
    assert confirmed["done_reason"]["value"] == "stop"
    # parâmetros de amostragem NUNCA aparecem como confirmados
    assert "temperature" not in confirmed
