"""Parsing de requisições e acumulação de streams (fixtures inline)."""

import json

from lombo.capture import StreamAccumulator, parse_request, tokens_per_second


def _ndjson(objs):
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def test_parse_request_api_chat():
    body = {"model": "llama3.2:3b", "stream": True,
            "options": {"temperature": 0.1, "top_k": 5},
            "messages": [{"role": "system", "content": "seja breve"},
                         {"role": "user", "content": "oi"}]}
    info = parse_request("/api/chat", body)
    assert info["model"] == "llama3.2:3b"
    assert info["stream"] is True
    assert info["requested_params"] == {"temperature": 0.1, "top_k": 5}
    assert info["requested_params_ollama"] == {"temperature": 0.1, "top_k": 5}
    assert info["system_prompt"] == "seja breve"


def test_parse_request_openai_maps_names_but_preserves_requested():
    body = {"model": "llama3.2:3b", "temperature": 0.2, "max_tokens": 50,
            "messages": [{"role": "user", "content": "oi"}]}
    info = parse_request("/v1/chat/completions", body)
    # camada solicitada preserva o nome enviado pelo cliente
    assert info["requested_params"] == {"temperature": 0.2, "max_tokens": 50}
    # tradução só para a camada inferida
    assert info["requested_params_ollama"] == {"temperature": 0.2, "num_predict": 50}
    assert info["stream"] is False  # OpenAI-compat: stream default False


def test_accumulate_chat_stream_with_thinking_separated():
    chunks = [
        {"model": "qwen3:4b", "message": {"role": "assistant", "thinking": "hmm, "}, "done": False},
        {"model": "qwen3:4b", "message": {"role": "assistant", "thinking": "2+2"}, "done": False},
        {"model": "qwen3:4b", "message": {"role": "assistant", "content": "A resposta "}, "done": False},
        {"model": "qwen3:4b", "message": {"role": "assistant", "content": "é 4."}, "done": False},
        {"model": "qwen3:4b", "message": {"role": "assistant", "content": ""}, "done": True,
         "done_reason": "stop", "total_duration": 5_000_000_000, "eval_count": 10,
         "eval_duration": 2_000_000_000, "prompt_eval_count": 7},
    ]
    acc = StreamAccumulator("/api/chat", stream=True)
    acc.feed(_ndjson(chunks))
    result = acc.result()
    assert result["response_text"] == "A resposta é 4."
    assert result["thinking_text"] == "hmm, 2+2"   # separado, nunca concatenado
    assert result["done_reason"] == "stop"
    assert result["metrics"]["eval_count"] == 10
    assert tokens_per_second(result["metrics"]) == 5.0
    assert result["parse_error"] is None


def test_accumulate_generate_nonstream():
    obj = {"model": "llama3.2:3b", "response": "olá!", "done": True,
           "done_reason": "stop", "eval_count": 3, "eval_duration": 1_000_000_000}
    acc = StreamAccumulator("/api/generate", stream=False)
    acc.feed(json.dumps(obj).encode())
    result = acc.result()
    assert result["response_text"] == "olá!"
    assert result["thinking_text"] is None
    assert result["metrics"]["eval_count"] == 3


def test_accumulate_openai_sse_stream():
    lines = []
    for delta in [{"content": "Oi"}, {"content": " tudo bem"}]:
        lines.append(b"data: " + json.dumps(
            {"choices": [{"delta": delta, "finish_reason": None}]}).encode() + b"\n\n")
    lines.append(b"data: " + json.dumps(
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 4}}).encode() + b"\n\n")
    lines.append(b"data: [DONE]\n\n")
    acc = StreamAccumulator("/v1/chat/completions", stream=True)
    acc.feed(b"".join(lines))
    result = acc.result()
    assert result["response_text"] == "Oi tudo bem"
    assert result["done_reason"] == "stop"
    assert result["metrics"] == {"prompt_eval_count": 5, "eval_count": 4}


def test_accumulate_garbage_reports_parse_error():
    acc = StreamAccumulator("/api/chat", stream=False)
    acc.feed(b"isso nao e json")
    result = acc.result()
    assert result["parse_error"] is not None
    assert result["response_text"] is None
