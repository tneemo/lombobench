"""Baseline do modelo e camadas de parâmetros.

Três camadas, mantidas SEMPRE separadas:
- solicitado: exatamente o que o cliente enviou;
- inferido: valor efetivo estimado pelo proxy (request > Modelfile > default do
  Ollama), cada item anotado com a origem e marcado como inferência;
- confirmado: somente o que a resposta daquela requisição confirma de fato.

O que vem de /api/ps é estado observado do runtime (observed_runtime_json),
nunca entra em `confirmed`.
"""

from __future__ import annotations

from typing import Any

# Defaults built-in do Ollama para os parâmetros de amostragem mais comuns.
# São INFERÊNCIAS dependentes da versão instalada — entram na camada inferida
# com origem 'ollama_default' e jamais na camada confirmada.
OLLAMA_BUILTIN_DEFAULTS: dict[str, Any] = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.0,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "num_predict": -1,
    "num_ctx": 4096,
    "seed": 0,
}

# Mapeamento dos nomes do endpoint OpenAI-compat para os nomes do Ollama.
# Usado apenas na camada INFERIDA; a camada solicitada preserva o nome enviado.
OPENAI_TO_OLLAMA = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "max_tokens": "num_predict",
    "max_completion_tokens": "num_predict",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "stop": "stop",
}


def parse_parameters_raw(text: str | None) -> dict[str, Any]:
    """Parseia o bloco `parameters` do POST /api/show (derivado; o bruto é preservado).

    Formato: uma linha por parâmetro, `nome<espaços>valor`. Chaves repetidas
    (ex.: vários `stop`) viram lista.
    """
    params: dict[str, Any] = {}
    if not text:
        return params
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, raw = parts
        value: Any = raw.strip().strip('"')
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                pass
        if key in params:
            if not isinstance(params[key], list):
                params[key] = [params[key]]
            params[key].append(value)
        else:
            params[key] = value
    return params


def layer_params(
    requested: dict[str, Any],
    modelfile_params: dict[str, Any],
    builtin_defaults: dict[str, Any] = OLLAMA_BUILTIN_DEFAULTS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Produz (inferred, diff, baseline_effective) a partir da camada solicitada.

    - inferred: {param: {"value": v, "source": "request"|"modelfile"|"ollama_default",
      "inferred": True}} para todo parâmetro conhecido por alguma das fontes.
    - diff: subconjunto em que o valor efetivo difere do baseline do modelo
      (Modelfile, ou default do Ollama na ausência dele), com valor base e origem.
    - baseline_effective: o valor base de cada parâmetro, para referência.

    Função pura (sem rede/IO) — coberta por testes.
    """
    inferred: dict[str, Any] = {}
    diff: dict[str, Any] = {}
    baseline_effective: dict[str, Any] = {}

    all_params = set(requested) | set(modelfile_params) | set(builtin_defaults)
    for param in sorted(all_params):
        if param in modelfile_params:
            base_value, base_source = modelfile_params[param], "modelfile"
        elif param in builtin_defaults:
            base_value, base_source = builtin_defaults[param], "ollama_default"
        else:
            base_value, base_source = None, None
        baseline_effective[param] = base_value

        if param in requested:
            value, source = requested[param], "request"
        else:
            value, source = base_value, base_source

        inferred[param] = {"value": value, "source": source, "inferred": True}
        if source == "request" and value != base_value:
            diff[param] = {
                "value": value,
                "base_value": base_value,
                "base_source": base_source,
            }
    return inferred, diff, baseline_effective


def confirmed_from_response(final_chunk: dict[str, Any] | None) -> dict[str, Any]:
    """Extrai APENAS o que a resposta desta requisição confirma explicitamente.

    O Ollama não ecoa parâmetros de amostragem na resposta; o que dá para
    confirmar honestamente é pouco (modelo servido, done_reason). A análise
    deve sempre tratar o resto como inferência — nunca promover.
    """
    confirmed: dict[str, Any] = {}
    if not final_chunk:
        return confirmed
    if final_chunk.get("model"):
        confirmed["model"] = {"value": final_chunk["model"], "source": "response"}
    if final_chunk.get("done_reason"):
        confirmed["done_reason"] = {"value": final_chunk["done_reason"], "source": "response"}
    return confirmed
