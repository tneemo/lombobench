"""Avaliações comportamentais heurísticas (source='heuristic').

Medem formato, repetição e desempenho da resposta observável. NÃO avaliam a
tese principal (comportamento epistêmico) — isso é papel dos rótulos humanos
aplicados no Lab (source='human').
"""

from __future__ import annotations

import json
import re
from typing import Any

# Rótulos epistêmicos válidos para avaliação humana no Lab.
HUMAN_LABELS = [
    "correto",
    "alucinou",
    "reconheceu_entidade_inexistente",
    "corrigiu_premissa_falsa",
    "pediu_esclarecimento",
    "demonstrou_incerteza",
    "resposta_inconclusiva",
]

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def evaluate(record: dict[str, Any], request_body: dict[str, Any] | None) -> list[tuple[str, Any]]:
    """Computa as métricas heurísticas de um registro concluído.

    Retorna pares (metric, value) para a tabela `evaluations`.
    """
    items: list[tuple[str, Any]] = []
    text = record.get("response_text") or ""
    thinking = record.get("thinking_text") or ""

    items.append(("response_chars", len(text)))
    words = _WORD_RE.findall(text.lower())
    items.append(("response_words", len(words)))

    if thinking:
        items.append(("thinking_chars", len(thinking)))
        total = len(text) + len(thinking)
        items.append(("thinking_ratio", round(len(thinking) / total, 4) if total else 0))

    if len(words) >= 6:
        trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        items.append(("repetition_rate", round(1 - len(set(trigrams)) / len(trigrams), 4)))
        items.append(("type_token_ratio", round(len(set(words)) / len(words), 4)))

    if record.get("tokens_per_second") is not None:
        items.append(("tokens_per_second", round(record["tokens_per_second"], 2)))
    if record.get("time_to_first_token_ms") is not None:
        items.append(("time_to_first_token_ms", round(record["time_to_first_token_ms"], 1)))

    if record.get("done_reason"):
        items.append(("truncated", record["done_reason"] in ("length", "max_tokens")))

    fmt = (request_body or {}).get("format")
    if fmt == "json" or isinstance(fmt, dict):
        items.append(("json_valid", _is_valid_json(text)))

    return items


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False
