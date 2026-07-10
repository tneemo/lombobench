"""Métricas comportamentais heurísticas."""

from lombo.behavior import HUMAN_LABELS, evaluate


def _metrics(record, body=None):
    return dict(evaluate(record, body or {}))


def test_basic_lengths_and_ratios():
    m = _metrics({"response_text": "um dois três um dois três um dois",
                  "thinking_text": None})
    assert m["response_chars"] == 33
    assert m["response_words"] == 8
    assert 0 < m["repetition_rate"] < 1   # trigrama "um dois três" repete
    assert 0 < m["type_token_ratio"] <= 1


def test_thinking_ratio_present_only_with_thinking():
    m = _metrics({"response_text": "resposta", "thinking_text": "raciocínio longo"})
    assert m["thinking_chars"] == 16
    assert 0 < m["thinking_ratio"] < 1
    assert "thinking_ratio" not in _metrics({"response_text": "resposta"})


def test_truncation_and_performance_passthrough():
    m = _metrics({"response_text": "ok", "done_reason": "length",
                  "tokens_per_second": 55.5555, "time_to_first_token_ms": 123.456})
    assert m["truncated"] is True
    assert m["tokens_per_second"] == 55.56
    assert m["time_to_first_token_ms"] == 123.5


def test_json_validity_only_when_format_requested():
    m = _metrics({"response_text": '{"a": 1}'}, {"format": "json"})
    assert m["json_valid"] is True
    m = _metrics({"response_text": "não é json"}, {"format": "json"})
    assert m["json_valid"] is False
    assert "json_valid" not in _metrics({"response_text": "não é json"}, {})


def test_human_labels_are_the_epistemic_set():
    assert "alucinou" in HUMAN_LABELS
    assert "corrigiu_premissa_falsa" in HUMAN_LABELS
    assert "reconheceu_entidade_inexistente" in HUMAN_LABELS
    assert len(HUMAN_LABELS) == 7
