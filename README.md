# LomboBench

Black-box observability of Ollama's **external inference controls** — phase 1 of LomboAI.

For every prompt sent to a model, LomboBench records which parameters were
**requested**, which effective values were **inferred** (and where they came
from: request, Modelfile, or Ollama default), what the server actually
**confirmed**, the diff from the model baseline, observable behavior (text,
separate thinking, metrics, time-to-first-token), and behavioral evaluations.
This data feeds phase 2 (an autonomous controller for these controls).

> **Honest framing:** this system does not access model weights, activations, or
> internal states — weights are immutable during inference. It observes external
> controls (sampling, template, system, context) and behavior. It does not
> replace mechanistic interpretability tools (NNsight, TransformerLens); phase 3
> will bridge that gap by rerunning recorded cases.

## Install

Requires Python 3.12+ and a running Ollama server at `127.0.0.1:11434`.

```sh
python -m venv .venv
```

Activate the virtual environment with the command for your platform:

```powershell
.venv\Scripts\Activate.ps1
```

```sh
source .venv/bin/activate
```

Then install and start LomboBench (the same commands on every platform):

```sh
pip install -e ".[dev]"
python -m lombo
```

Open <http://127.0.0.1:11435/_lombo/>. On Windows, the virtual environment's
Python executable is `.venv\Scripts\python.exe`.

## Architecture

```
Any client (curl, LangChain, Open WebUI…)
        │  http://127.0.0.1:11435  (drop-in: same API as Ollama)
        ▼
   LomboProxy (FastAPI + httpx)
        │── forwards everything to Ollama :11434 (transparent streaming)
        │── captures /api/generate, /api/chat, /v1/chat/completions
        │── Lab + analysis at /_lombo/
        ▼
   lombo.db (SQLite, single writer via queue)
```

## Usage

```sh
python -m lombo            # proxy on :11435, Lab at /_lombo/
# flags: --port --host --ollama-url --db --no-capture
```

Point any client to `http://127.0.0.1:11435` instead of Ollama.
Experimental metadata via headers (removed before forwarding):
`X-Lombo-Experiment`, `X-Lombo-Condition`, `X-Lombo-Pair`, `X-Lombo-Replicate`,
`X-Lombo-No-Log` (does not record the request).

### Inference engines

The `lombo.engines.Engine` contract separates the benchmark from the runtime.
It exposes `generate`, `tokenize`, `get_model_info`, and the `supports_trace`
hook; results use the same normalized fields as current capture. The existing
Ollama proxy remains the default path and has not been changed.

For local inference with Hugging Face/PyTorch weights (not GGUF):

```sh
pip install -e ".[transformers]"
```

```python
from lombo.engines import TransformersEngine

engine = TransformersEngine("Qwen/Qwen2.5-0.5B-Instruct")
result = engine.generate(
    [{"role": "user", "content": "Explain entropy in one sentence."}],
    seed=42, temperature=0.7, top_p=0.9, top_k=40, max_new_tokens=80,
)
print(result["response_text"])
```

The engine selects devices with `device_map="auto"`, uses BF16 when the GPU
supports it (FP16 on other GPUs and FP32 on CPU), and loads heavy dependencies
only when instantiated.

Initial opt-in support for NNsight is installed separately:

```sh
pip install -e ".[transformers,nnsight]"
```

```python
# single-layer trace (NNsight opt-in):
if engine.supports_trace():
    trace = engine.trace("Explain entropy.", layer_index=0)

# multi-layer correlational diagnostic (logit lens, without NNsight):
if engine.supports_diagnostics():
    diag = engine.trace_all_layers("Explain entropy.", target_token="order")
```

`trace` runs a single forward pass and captures only the output of the specified
layer and the logits at the final position. The serializable return contains
the layer index, activation shape and dtype, token IDs, final-token logits, and
model metadata.

`trace_all_layers` (layer diagnostics, exposed in the Lab under the
**Diagnostics** tab) performs the **logit lens** in one **read-only** pass — a
forward hook on each block captures the residual, projected through the final
norm and `lm_head` (it does not use NNsight; the read is passive). For each
layer it reports the top-k tokens, hidden-state norms (‖h‖ and residual Δ), and
— if a target token is provided — its rank and probability. Projecting the final
layer's residual reproduces the model's actual logits, anchoring the lens to the
true prediction. This shows **at which layer a prediction becomes *legible***
(representational readout), which is **not** the same as proving that the layer
*computes* or *uses* the feature. A causal answer (“used”) requires intervention
— activation patching, where NNsight excels — planned for the next phase.
Attention, MLP, cache, ablation, patching, and SAEs remain out of scope for this
phase.

### Lombo Lab (`http://127.0.0.1:11435/_lombo/`)

Side-by-side chat panels with **one variable manipulated per panel**
(temperature, seed, model, system…). Two modes per session:

- **Controlled experiment (default)** — each round is independent: all panels
  receive the same base history and only the variable differs. Comparable; this
  is the paper's mode.
- **Branched conversation** — each panel follows its own history
  (`branch_id`/`parent_request_id` record the tree).

Each submission generates a `pair_id`; resubmissions increment `replicate`.
Without a defined seed, the Lab injects an explicit seed per round (the same in
every panel) — every record is **replayable** (hook for phase 3's “Inspect
execution”). Under each response: word-by-word diff, metrics, and human epistemic
labels (*correct, hallucinated, recognized a nonexistent entity, corrected a
false premise, asked for clarification, expressed uncertainty, inconclusive
response*), stored in `evaluations` with `source='human'`.

### Analysis and export

**Analysis** tab (or `/_lombo/api/*`): timeline, detail with the three parameter
layers (never merged — `/api/ps` is kept separately in `observed_runtime_json`),
`GET /summary`, `GET /compare?pair_id=`,
`GET /params/impact?param=temperature`, `GET /export?format=csv|json`.

## Proxy overhead

Measured, not claimed: `python scripts/overhead_bench.py`. On the development
machine (Ollama 0.31.2, llama3.2:3b, n=8), the median difference was within
measurement noise (−10.8 ms total; −4.6 ms TTFT).

## Privacy

The proxy records prompts and responses locally. By default it listens **only
on 127.0.0.1**; `X-Lombo-No-Log` excludes a request; `--no-capture` disables
capture; the Lab displays a permanent warning.

## Development

```sh
python -m pytest    # parameter layers, stream parsing, heuristics
```

Out of scope for this phase (prepared in the schema): autonomous controller
(phase 2), per-token logprobs, model-judge evaluations. Interpretability via
NNsight (phase 3) already provides **correlational** layer diagnostics
(multi-layer logit lens, Diagnostics tab); **causal** diagnostics via activation
patching are the next step (`requests.request_body_raw` + seed + digest guarantee
replay).

## Docs

- [Design language](docs/DESIGN_LANGUAGE.md)
- [Thinking-to-Response Transfer](docs/notes/thinking-to-response-transfer.md)

## Third-party

`lombo/static/vendor/echarts.min.js` is Apache ECharts, licensed under Apache-2.0
(see [`lombo/static/vendor/LICENSE-echarts`](lombo/static/vendor/LICENSE-echarts)).

## License

MIT; see [LICENSE](LICENSE).
