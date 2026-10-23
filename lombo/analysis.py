"""API de análise (/_lombo/api/*): listagem, detalhe, agregações, export.

Valores de parâmetros são sempre apresentados com sua camada (solicitado /
inferido / confirmado) — a análise nunca funde as camadas.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import db
from .behavior import HUMAN_LABELS

router = APIRouter()

_JSON_FIELDS = ("requested_params_json", "inferred_params_json", "confirmed_params_json",
                "observed_runtime_json", "params_diff_json", "prompt_or_messages_json",
                "regulator_action_json")

_LIST_COLUMNS = ("id, started_at, completed_at, endpoint, model, origin, lab_session_id, "
                 "turn_id, branch_id, experiment, condition, pair_id, replicate, "
                 "params_diff_json, status_code, error, eval_count, tokens_per_second, "
                 "time_to_first_token_ms, done_reason")


def _row_to_dict(row: sqlite3.Row, parse_json: bool = False) -> dict:
    d = dict(row)
    if parse_json:
        for field in _JSON_FIELDS:
            if d.get(field):
                try:
                    d[field.removesuffix("_json")] = json.loads(d.pop(field))
                except ValueError:
                    pass
    return d


@router.get("/requests")
async def list_requests(request: Request, model: str | None = None,
                        experiment: str | None = None, condition: str | None = None,
                        pair_id: str | None = None, session: str | None = None,
                        origin: str | None = None, limit: int = 100, offset: int = 0):
    conn = request.app.state.conn
    clauses, args = [], []
    for column, value in [("model", model), ("experiment", experiment),
                          ("condition", condition), ("pair_id", pair_id),
                          ("lab_session_id", session), ("origin", origin)]:
        if value is not None:
            clauses.append(f"{column} = ?")
            args.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT {_LIST_COLUMNS} FROM requests {where} "
        f"ORDER BY started_at DESC LIMIT ? OFFSET ?", (*args, limit, offset)
    ).fetchall()
    return {"requests": [_row_to_dict(r, parse_json=True) for r in rows]}


@router.get("/requests/{request_id}")
async def get_request(request: Request, request_id: str):
    conn = request.app.state.conn
    row = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "request não encontrada")
    result = _row_to_dict(row, parse_json=True)
    evals = conn.execute(
        "SELECT metric, value, source, created_at FROM evaluations WHERE request_id = ?",
        (request_id,)).fetchall()
    result["evaluations"] = [
        {**dict(e), "value": _loads_or_raw(e["value"])} for e in evals]
    if row["environment_id"]:
        env = conn.execute("SELECT * FROM environments WHERE id = ?",
                           (row["environment_id"],)).fetchone()
        result["environment"] = dict(env) if env else None
    if row["model"]:
        base = db.get_baseline(conn, row["model"])
        if base:
            result["baseline"] = {k: base[k] for k in
                                  ("model", "digest", "captured_at", "parameters_raw",
                                   "template_text", "system_text")}
    return result


class HumanEvaluation(BaseModel):
    labels: list[str]
    note: str | None = None


class TraceRequest(BaseModel):
    layer_index: int


class DiagnosticsRequest(BaseModel):
    target_token: str | None = None
    top_k: int = 10


class AttentionRequest(BaseModel):
    layer_index: int = 0
    head_index: int | None = None


class ActivationsRequest(BaseModel):
    position: int = -1


class ActivationsInspectRequest(BaseModel):
    layer_index: int
    module: str
    position: int = -1
    top_k_dims: int = 16
    bins: int = 32


def _resolve_trace_engine(state, request_id: str):
    """Shared guards for the NNsight endpoints: returns (engine, prompt).

    Raises the same HTTPExceptions as the single-layer trace: 404 (no request),
    409 (engine ≠ transformers or model not loaded), 503 (NNsight unavailable),
    422 (no prompt to re-run).
    """
    conn = state.conn
    row = conn.execute(
        "SELECT model, lab_session_id, prompt_or_messages_json FROM requests WHERE id = ?",
        (request_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "request não encontrada")

    engine_name = "ollama"
    if row["lab_session_id"]:
        session = conn.execute(
            "SELECT base_config_json FROM lab_sessions WHERE id = ?",
            (row["lab_session_id"],),
        ).fetchone()
        if session:
            engine_name = json.loads(session["base_config_json"] or "{}").get(
                "engine", "ollama")
    if engine_name != "transformers":
        raise HTTPException(409, "NNsight direto está disponível somente para Transformers")

    engine = state.transformer_engines.get(row["model"])
    if engine is None:
        raise HTTPException(409, "modelo Transformers não está carregado; envie uma mensagem primeiro")

    prompt = json.loads(row["prompt_or_messages_json"] or "null")
    if prompt is None:
        raise HTTPException(422, "request sem prompt ou mensagens para reexecutar")
    return engine, prompt


@router.post("/requests/{request_id}/trace")
async def trace_request(request: Request, request_id: str, body: TraceRequest):
    """Run the minimal opt-in NNsight trace for a Transformers Lab request."""
    engine, prompt = _resolve_trace_engine(request.app.state, request_id)
    if not engine.supports_trace():
        raise HTTPException(503, "NNsight não está instalado ou o modelo não é compatível")
    try:
        result = await asyncio.to_thread(engine.trace, prompt, body.layer_index)
    except (IndexError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"trace NNsight falhou: {exc}") from exc
    return result


@router.post("/requests/{request_id}/diagnostics")
async def diagnostics_request(request: Request, request_id: str, body: DiagnosticsRequest):
    """Correlational multi-layer diagnostic (logit lens) for a Transformers request.

    Representational readout: WHERE a prediction becomes legible per layer — not
    causal evidence that the layer computes/uses the feature. Read-only forward
    pass; no NNsight required (that is reserved for the causal patching phase).
    """
    engine, prompt = _resolve_trace_engine(request.app.state, request_id)
    if not engine.supports_diagnostics():
        raise HTTPException(422, "arquitetura do modelo não suportada para diagnóstico de camadas")
    try:
        result = await asyncio.to_thread(
            engine.trace_all_layers, prompt, body.target_token, body.top_k)
    except (IndexError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"diagnóstico de camadas falhou: {exc}") from exc
    return result


@router.post("/requests/{request_id}/attention")
async def attention_request(request: Request, request_id: str, body: AttentionRequest):
    """Real attention weights (token×token) for one layer/head of a Transformers request.

    Internal quantities read from a forward pass with ``output_attentions``. They
    show WHERE each query position sends its attention mass — suggestive of
    information routing, not causal proof that the attended value is used. No
    NNsight required; the causal story (patching) stays in the later phase.
    """
    engine, prompt = _resolve_trace_engine(request.app.state, request_id)
    if not engine.supports_attention():
        raise HTTPException(422, "arquitetura do modelo não suportada para mapa de atenção")
    try:
        result = await asyncio.to_thread(
            engine.attention, prompt, body.layer_index, body.head_index)
    except (IndexError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"mapa de atenção falhou: {exc}") from exc
    return result


@router.post("/requests/{request_id}/activations")
async def activations_request(request: Request, request_id: str, body: ActivationsRequest):
    """Per-layer, per-module activation summary (norm/mean/max/sparsity) for a
    Transformers request — the "overview" matrix. See ``.../activations/inspect``
    for the per-dimension detail of a single cell."""
    engine, prompt = _resolve_trace_engine(request.app.state, request_id)
    if not engine.supports_activations():
        raise HTTPException(422, "arquitetura do modelo não suportada para captura de ativações")
    try:
        result = await asyncio.to_thread(engine.capture_activations, prompt, body.position)
    except (IndexError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"captura de ativações falhou: {exc}") from exc
    return result


@router.post("/requests/{request_id}/activations/inspect")
async def activations_inspect_request(request: Request, request_id: str,
                                      body: ActivationsInspectRequest):
    """Per-dimension detail (top outliers + histogram) of one activation cell."""
    engine, prompt = _resolve_trace_engine(request.app.state, request_id)
    if not engine.supports_activations():
        raise HTTPException(422, "arquitetura do modelo não suportada para captura de ativações")
    if body.module not in ("attn", "mlp", "resid"):
        raise HTTPException(422, "module deve ser um de: attn, mlp, resid")
    try:
        result = await asyncio.to_thread(
            engine.inspect_activation, prompt, body.layer_index, body.module,
            body.position, body.top_k_dims, body.bins)
    except (IndexError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"inspeção de ativação falhou: {exc}") from exc
    return result


@router.post("/requests/{request_id}/evaluations")
async def add_human_evaluation(request: Request, request_id: str, body: HumanEvaluation):
    """Rótulos epistêmicos aplicados por humano no Lab (source='human')."""
    conn = request.app.state.conn
    if conn.execute("SELECT 1 FROM requests WHERE id = ?", (request_id,)).fetchone() is None:
        raise HTTPException(404, "request não encontrada")
    invalid = [label for label in body.labels if label not in HUMAN_LABELS]
    if invalid:
        raise HTTPException(422, f"rótulos inválidos: {invalid}; válidos: {HUMAN_LABELS}")
    created_at = datetime.now(timezone.utc).isoformat()
    items = [("label", {"label": label, "note": body.note}) for label in body.labels]
    request.app.state.writer.enqueue(
        lambda conn: db.insert_evaluations(conn, request_id, items, "human", created_at))
    return {"ok": True, "labels": body.labels}


@router.get("/labels")
async def labels():
    return {"labels": HUMAN_LABELS}


@router.get("/summary")
async def summary(request: Request, experiment: str | None = None):
    conn = request.app.state.conn
    where, args = ("WHERE experiment = ?", [experiment]) if experiment else ("", [])
    by_model = conn.execute(
        f"SELECT model, condition, COUNT(*) AS n, AVG(tokens_per_second) AS avg_tps, "
        f"AVG(time_to_first_token_ms) AS avg_ttft_ms, AVG(eval_count) AS avg_eval_count "
        f"FROM requests {where} GROUP BY model, condition ORDER BY model", args).fetchall()
    return {"groups": [dict(r) for r in by_model]}


@router.get("/compare")
async def compare(request: Request, pair_id: str):
    conn = request.app.state.conn
    rows = conn.execute(
        "SELECT * FROM requests WHERE pair_id = ? ORDER BY condition", (pair_id,)).fetchall()
    out = []
    for row in rows:
        d = _row_to_dict(row, parse_json=True)
        evals = conn.execute(
            "SELECT metric, value, source FROM evaluations WHERE request_id = ?",
            (row["id"],)).fetchall()
        d["evaluations"] = [{**dict(e), "value": _loads_or_raw(e["value"])} for e in evals]
        if row["model"]:
            base = db.get_baseline(conn, row["model"])
            d["model_digest"] = base["digest"] if base else None
        out.append(d)
    return {"pair_id": pair_id, "requests": out}


@router.get("/params/impact")
async def params_impact(request: Request, param: str, model: str | None = None):
    """Agrupa requisições pelo valor INFERIDO do parâmetro e compara métricas.

    O agrupamento usa a camada inferida (com origem explícita) — o resultado
    indica para cada grupo se o valor veio de request/modelfile/default.
    """
    conn = request.app.state.conn
    where, args = ("AND model = ?", [model]) if model else ("", [])
    rows = conn.execute(
        f"SELECT inferred_params_json, tokens_per_second, eval_count, "
        f"time_to_first_token_ms FROM requests WHERE inferred_params_json IS NOT NULL {where}",
        args).fetchall()
    groups: dict[str, dict] = {}
    for row in rows:
        info = json.loads(row["inferred_params_json"]).get(param)
        if info is None:
            continue
        key = json.dumps(info["value"])
        g = groups.setdefault(key, {"value": info["value"], "sources": set(), "n": 0,
                                    "tps": [], "eval_count": [], "ttft_ms": []})
        g["sources"].add(info["source"])
        g["n"] += 1
        for field, col in [("tps", "tokens_per_second"), ("eval_count", "eval_count"),
                           ("ttft_ms", "time_to_first_token_ms")]:
            if row[col] is not None:
                g[field].append(row[col])
    return {"param": param, "note": "valores da camada INFERIDA (não confirmada pelo servidor)",
            "groups": [{"value": g["value"], "sources": sorted(g["sources"]), "n": g["n"],
                        "avg_tokens_per_second": _avg(g["tps"]),
                        "avg_eval_count": _avg(g["eval_count"]),
                        "avg_ttft_ms": _avg(g["ttft_ms"])} for g in groups.values()]}


@router.get("/export")
async def export(request: Request, format: str = "json"):
    conn = request.app.state.conn
    rows = conn.execute("SELECT * FROM requests ORDER BY started_at").fetchall()
    evals = conn.execute("SELECT * FROM evaluations").fetchall()
    if format == "csv":
        buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(dict(r) for r in rows)
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=lombo_requests.csv"})
    return {"requests": [_row_to_dict(r) for r in rows],
            "evaluations": [dict(e) for e in evals]}


def _avg(values: list) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _loads_or_raw(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value
