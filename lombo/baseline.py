"""Model baseline and parameter layers.

Three layers, ALWAYS kept separate:
- requested: exactly what the client sent;
- inferred: effective value estimated by the proxy (request > Modelfile > Ollama
  default), each item annotated with its source and marked as an inference;
- confirmed: only what that request's response actually confirms.

Data from /api/ps is observed runtime state (observed_runtime_json),
and never enters `confirmed`.
"""

from __future__ import annotations

from typing import Any

# Ollama built-in defaults for the most common sampling parameters.
# These are INFERENCES dependent on the installed version — they enter the inferred layer
# with source 'ollama_default' and never the confirmed layer.
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

# Map OpenAI-compatible endpoint names to Ollama names.
# Used only in the INFERRED layer; the requested layer preserves the submitted name.
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
    """Parse the `parameters` block from POST /api/show (derived; raw data is preserved).

    Format: one line per parameter, `name<spaces>value`. Repeated keys
    (e.g. multiple `stop` entries) become a list.
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
    """Produce (inferred, diff, baseline_effective) from the requested layer.

    - inferred: {param: {"value": v, "source": "request"|"modelfile"|"ollama_default",
      "inferred": True}} for every parameter known to any source.
    - diff: subset where the effective value differs from the model baseline
      (Modelfile, or Ollama default if absent), with baseline value and source.
    - baseline_effective: the baseline value of each parameter, for reference.

    Pure function (no network/IO) — covered by tests.
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
    """Extract ONLY what this request's response explicitly confirms.

    Ollama does not echo sampling parameters in the response; little can be
    honestly confirmed (served model, done_reason). Analysis must always
    treat the rest as inference — never promote it.
    """
    confirmed: dict[str, Any] = {}
    if not final_chunk:
        return confirmed
    if final_chunk.get("model"):
        confirmed["model"] = {"value": final_chunk["model"], "source": "response"}
    if final_chunk.get("done_reason"):
        confirmed["done_reason"] = {"value": final_chunk["done_reason"], "source": "response"}
    return confirmed
