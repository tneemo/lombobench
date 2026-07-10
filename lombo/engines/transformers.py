"""PyTorch/Hugging Face Transformers inference engine."""

from __future__ import annotations

import time
import threading
from typing import Any

from .base import Engine, GenerationResult


class TransformersEngine(Engine):
    def __init__(self, model: str, **model_kwargs: Any) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise ImportError(
                "TransformersEngine requires the 'transformers' extra: "
                "pip install 'lombo[transformers]'"
            ) from exc

        self.model_name = model
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        load_kwargs = {"device_map": "auto", "torch_dtype": self._best_dtype(), **model_kwargs}
        self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        self.model.eval()
        self._generation_lock = threading.Lock()

    def _best_dtype(self):
        torch = self._torch
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def generate(self, prompt: str | list[dict[str, Any]], **params: Any) -> GenerationResult:
        torch = self._torch
        text = self._format_prompt(prompt)
        encoded = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

        seed = int(params.pop("seed", 0))
        generation = self._generation_params(params)
        started = time.perf_counter_ns()
        try:
            # Transformers 5.x no longer accepts ``generator`` as a generate
            # kwarg. Seed the PyTorch RNG while holding a per-model lock so
            # parallel benchmark panes cannot overwrite each other's seed.
            with self._generation_lock:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    output = self.model.generate(**encoded, **generation)
            duration = time.perf_counter_ns() - started
            prompt_tokens = encoded["input_ids"].shape[-1]
            new_ids = output[0, prompt_tokens:]
            response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            count = int(new_ids.shape[-1])
            eos = self.model.generation_config.eos_token_id
            last = int(new_ids[-1]) if count else None
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            done_reason = "stop" if last in eos_ids else "length"
            final = {
                "model": self.model_name,
                "done": True,
                "done_reason": done_reason,
                "prompt_eval_count": int(prompt_tokens),
                "eval_count": count,
                "eval_duration": duration,
            }
            return {
                "response_text": response or None,
                "thinking_text": None,
                "final_chunk": final,
                "metrics": {k: final[k] for k in
                            ("prompt_eval_count", "eval_count", "eval_duration")},
                "done_reason": done_reason,
                "parse_error": None,
            }
        except Exception as exc:
            return {"response_text": None, "thinking_text": None, "final_chunk": None,
                    "metrics": {}, "done_reason": None,
                    "parse_error": f"{type(exc).__name__}: {exc}"}

    def _format_prompt(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            return prompt
        if not hasattr(self.tokenizer, "apply_chat_template") or \
                getattr(self.tokenizer, "chat_template", None) is None:
            # No chat template configured (e.g. base gpt2): fall back to a
            # plain-text transcript instead of failing outright.
            lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt]
            return "\n".join(lines) + "\nassistant:"
        return self.tokenizer.apply_chat_template(prompt, tokenize=False,
                                                  add_generation_prompt=True)

    @staticmethod
    def _generation_params(params: dict[str, Any]) -> dict[str, Any]:
        aliases = {"num_predict": "max_new_tokens", "max_tokens": "max_new_tokens",
                   "repeat_penalty": "repetition_penalty"}
        allowed = {"temperature", "top_p", "top_k", "max_new_tokens",
                   "repetition_penalty", "do_sample", "num_beams"}
        mapped = {aliases.get(key, key): value for key, value in params.items()}
        result = {key: value for key, value in mapped.items() if key in allowed}
        if "do_sample" not in result:
            result["do_sample"] = bool(result.get("temperature", 1.0) > 0)
        if result.get("temperature") == 0:
            result.pop("temperature")
        return result

    def tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def get_model_info(self) -> dict[str, Any]:
        return {
            "engine": "transformers",
            "model": self.model_name,
            "architecture": type(self.model).__name__,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
        }

    def supports_trace(self) -> bool:
        """Return true only for installed NNsight and a known causal-LM layout."""
        try:
            from nnsight import LanguageModel  # noqa: F401
        except (ImportError, AttributeError):
            return False
        return self._layer_path() is not None and hasattr(self.model, "lm_head")

    def trace(self, prompt: str | list[dict[str, Any]], layer_index: int) -> dict[str, Any]:
        """Capture one layer output and final-position logits in one forward pass."""
        if not self.supports_trace():
            raise RuntimeError(
                "NNsight is unavailable or this model architecture is not supported"
            )

        from nnsight import LanguageModel

        layers, path = self._layer_path()  # type: ignore[misc]
        if not -len(layers) <= layer_index < len(layers):
            raise IndexError(
                f"layer_index {layer_index} is out of range for {len(layers)} layers"
            )
        normalized_index = layer_index % len(layers)
        text = self._format_prompt(prompt)
        encoded = self.tokenizer(text, return_tensors="pt")
        token_ids = encoded["input_ids"][0].tolist()

        traced = LanguageModel(self.model, tokenizer=self.tokenizer)
        traced_layers = self._resolve_path(traced, path)
        with traced.trace(encoded):
            # Supported causal-LM blocks return a tuple whose first item is
            # the hidden state. Save exactly this layer, not every layer.
            activation = traced_layers[normalized_index].output[0].save()
            last_logits = traced.lm_head.output[0, -1].save()

        activation = self._materialize_tensor(activation)
        last_logits = self._materialize_tensor(last_logits)
        return {
            "layer_index": normalized_index,
            "activation_shape": list(activation.shape),
            "activation_dtype": str(activation.dtype),
            "tokens": [int(token) for token in token_ids],
            "last_token_logits": last_logits.tolist(),
            "model": self.get_model_info(),
        }

    def supports_diagnostics(self) -> bool:
        """Return true when the logit-lens diagnostic can run on this model.

        Unlike ``supports_trace`` this needs no NNsight: the logit lens is a
        read-only forward pass. It only requires a known decoder-layer layout
        (to hook every block) and an ``lm_head`` (to project residuals).
        """
        return self._layer_path() is not None and hasattr(self.model, "lm_head")

    def trace_all_layers(
        self,
        prompt: str | list[dict[str, Any]],
        target_token: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Correlational multi-layer diagnostic (logit lens) in one forward pass.

        A forward hook on every decoder block captures its last-token residual;
        each residual is projected through the model's final norm and ``lm_head``
        to report, per layer, the top predicted tokens, hidden-state norms and
        (optionally) the rank/probability of a target token. Projecting the last
        layer's residual reproduces the model's real logits exactly, so the lens
        is anchored to the true prediction.

        This measures WHERE a prediction becomes *readable*, not where a feature
        is causally computed/used. Causal localisation (activation patching, the
        NNsight strength) is a separate, heavier method left to a later phase.
        No NNsight is involved here — the lens is a plain read-only pass.
        """
        if not self.supports_diagnostics():
            raise RuntimeError(
                "this model architecture is not supported for layer diagnostics"
            )

        torch = self._torch
        layers, _ = self._layer_path()  # type: ignore[misc]
        norm_path = self._final_norm_path()
        final_norm = self._resolve_path(self.model, norm_path) if norm_path else None
        lm_head = self.model.lm_head

        text = self._format_prompt(prompt)
        encoded = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        token_ids = encoded["input_ids"][0].tolist()

        target_token_id = None
        if target_token:
            # Resolve the target to a single leading token; a leading space is
            # how most BPE/SentencePiece tokenizers encode a mid-sentence word.
            ids = self.tokenizer.encode(" " + target_token, add_special_tokens=False) \
                or self.tokenizer.encode(target_token, add_special_tokens=False)
            target_token_id = int(ids[0]) if ids else None

        # One forward pass; a hook per decoder block records its residual output.
        residuals: list[Any] = []

        def _capture(_module, _inputs, output):
            residuals.append(output[0] if isinstance(output, tuple) else output)

        handles = [layer.register_forward_hook(_capture) for layer in layers]
        try:
            with torch.inference_mode():
                self.model(**encoded, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        layers_out = []
        prev_hidden = None
        with torch.inference_mode():
            for index, residual in enumerate(residuals):
                last = residual[0, -1]
                hidden = last.detach().float()
                normed = final_norm(last) if final_norm is not None else last
                logits = lm_head(normed).detach().float()
                probs = torch.softmax(logits, dim=-1)
                top = torch.topk(probs, min(top_k, int(probs.shape[-1])))
                entry: dict[str, Any] = {
                    "index": index,
                    "top": [
                        {"token_id": int(tok), "token": self.tokenizer.decode([int(tok)]),
                         "prob": float(prob)}
                        for tok, prob in zip(top.indices.tolist(), top.values.tolist())
                    ],
                    "hidden_norm": float(hidden.norm()),
                    "delta_norm": 0.0 if prev_hidden is None
                                  else float((hidden - prev_hidden).norm()),
                    "target_rank": None,
                    "target_prob": None,
                }
                if target_token_id is not None:
                    target_prob = probs[target_token_id]
                    entry["target_prob"] = float(target_prob)
                    # rank 1 == highest probability at this layer
                    entry["target_rank"] = int((probs > target_prob).sum()) + 1
                layers_out.append(entry)
                prev_hidden = hidden

        return {
            "num_layers": len(layers_out),
            "target_token": target_token,
            "target_token_id": target_token_id,
            "final_norm_applied": final_norm is not None,
            "tokens": [int(token) for token in token_ids],
            "layers": layers_out,
            "model": self.get_model_info(),
        }

    def _layer_path(self):
        """Return (layers, attribute path) for the few layouts we support."""
        candidates = (
            ("model", "layers"),       # Llama, Mistral, Qwen and peers
            ("transformer", "h"),      # GPT-2 style
            ("gpt_neox", "layers"),   # GPT-NeoX/Pythia style
        )
        for path in candidates:
            try:
                layers = self._resolve_path(self.model, path)
                if len(layers) > 0:
                    return layers, path
            except (AttributeError, TypeError):
                continue
        return None

    def _final_norm_path(self):
        """Return the attribute path to the final norm for the logit lens.

        Mirrors ``_layer_path``'s known layouts. Returns ``None`` when the norm
        cannot be located, in which case the lens projects the raw residual —
        an honest degradation the payload flags via ``final_norm_applied``.
        """
        candidates = (
            ("model", "norm"),               # Llama, Mistral, Qwen and peers
            ("transformer", "ln_f"),         # GPT-2 style
            ("gpt_neox", "final_layer_norm"),  # GPT-NeoX/Pythia style
        )
        for path in candidates:
            try:
                if self._resolve_path(self.model, path) is not None:
                    return path
            except (AttributeError, TypeError):
                continue
        return None

    @staticmethod
    def _resolve_path(root: Any, path: tuple[str, ...]) -> Any:
        value = root
        for name in path:
            value = getattr(value, name)
        return value

    @staticmethod
    def _materialize_tensor(value: Any) -> Any:
        """Detach saved tensors without assuming whether NNsight used a proxy."""
        if hasattr(value, "value"):
            value = value.value
        return value.detach().cpu()
