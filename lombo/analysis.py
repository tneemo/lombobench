"""API de análise (/_lombo/api/*): listagem, detalhe, agregações, export.

Valores de parâmetros são sempre apresentados com sua camada (solicitado /
inferido / confirmado) — a análise nunca funde as camadas.
"""

from __future__ import annotations

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


@router.post("/requests/{request_id}/evaluations")
async def add_human_evaluation(request: Request, request_id: str, body: HumanEvaluation):
    """Rótulos epistêmicos aplicados por humano no Lab (source='human')."""
    conn = request.app.state.conn
    if conn.execute("SELECT 1 FROM requests WHERE id = ?", (request_id,)).fetchone() is None:
        raise HTTPException(404, "request não encontrada")
    invalid = [l for l in body.labels if l not in HUMAN_LABELS]
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
