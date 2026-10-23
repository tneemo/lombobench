"""Aggregation API (/_lombo/api/*): time series, latency, distributions,
local model catalog, environments, aggregated evaluations and VRAM state.

Everything here only reads data already recorded by analysis.py/proxy.py —
no endpoint writes, no existing shape changes. The same separate layers
(requested/inferred/confirmed) apply here: aggregations are always labeled
with the source of the data they summarize.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()

_DISTRIBUTION_METRICS = {"tokens_per_second", "time_to_first_token_ms", "eval_count"}


@router.get("/metrics/timeseries")
async def metrics_timeseries(request: Request, model: str | None = None,
                             experiment: str | None = None, origin: str | None = None,
                             bucket: str = "hour", limit: int = 500):
    if bucket not in ("hour", "day"):
        raise HTTPException(422, "bucket must be 'hour' or 'day'")
    conn = request.app.state.conn
    length = 13 if bucket == "hour" else 10
    clauses, args = [], []
    for column, value in [("model", model), ("experiment", experiment), ("origin", origin)]:
        if value is not None:
            clauses.append(f"{column} = ?")
            args.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT substr(started_at, 1, {length}) AS t, COUNT(*) AS n, "
        f"SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors, "
        f"AVG(tokens_per_second) AS avg_tps, AVG(time_to_first_token_ms) AS avg_ttft_ms, "
        f"SUM(eval_count) AS total_tokens "
        f"FROM requests {where} GROUP BY t ORDER BY t DESC LIMIT ?", (*args, limit)).fetchall()
    points = [{"t": r["t"], "n": r["n"], "errors": r["errors"],
              "avg_tps": _round_or_none(r["avg_tps"]),
              "avg_ttft_ms": _round_or_none(r["avg_ttft_ms"]),
              "total_tokens": r["total_tokens"] or 0} for r in rows]
    points.reverse()
    return {"bucket": bucket, "points": points}


@router.get("/metrics/latency")
async def metrics_latency(request: Request, model: str | None = None,
                          experiment: str | None = None, limit: int = 500):
    conn = request.app.state.conn
    clauses, args = ["status_code = 200"], []
    for column, value in [("model", model), ("experiment", experiment)]:
        if value is not None:
            clauses.append(f"{column} = ?")
            args.append(value)
    where = f"WHERE {' AND '.join(clauses)}"
    group_rows = conn.execute(
        f"SELECT model, COUNT(*) AS n, AVG(load_duration_ns) AS avg_load_ns, "
        f"AVG(prompt_eval_duration_ns) AS avg_prompt_eval_ns, "
        f"AVG(eval_duration_ns) AS avg_eval_ns, AVG(time_to_first_token_ms) AS avg_ttft_ms, "
        f"AVG(prompt_eval_count) AS avg_prompt_eval_count "
        f"FROM requests {where} GROUP BY model", args).fetchall()
    point_rows = conn.execute(
        f"SELECT id, started_at, model, load_duration_ns, prompt_eval_duration_ns, "
        f"eval_duration_ns, time_to_first_token_ms, eval_count, prompt_eval_count, "
        f"tokens_per_second FROM requests {where} ORDER BY started_at DESC LIMIT ?",
        (*args, limit)).fetchall()
    groups = [{"model": r["model"], "n": r["n"],
              "avg_load_ms": _ns_to_ms(r["avg_load_ns"]),
              "avg_prompt_eval_ms": _ns_to_ms(r["avg_prompt_eval_ns"]),
              "avg_eval_ms": _ns_to_ms(r["avg_eval_ns"]),
              "avg_ttft_ms": _round_or_none(r["avg_ttft_ms"]),
              "avg_prompt_eval_count": _round_or_none(r["avg_prompt_eval_count"], 1)}
             for r in group_rows]
    points = [{"id": r["id"], "started_at": r["started_at"], "model": r["model"],
              "load_ms": _ns_to_ms(r["load_duration_ns"]),
              "prompt_eval_ms": _ns_to_ms(r["prompt_eval_duration_ns"]),
              "eval_ms": _ns_to_ms(r["eval_duration_ns"]),
              "ttft_ms": r["time_to_first_token_ms"], "eval_count": r["eval_count"],
              "prompt_eval_count": r["prompt_eval_count"],
              "tokens_per_second": r["tokens_per_second"]} for r in point_rows]
    return {"groups": groups, "points": points}


@router.get("/metrics/distributions")
async def metrics_distributions(request: Request, metric: str, model: str | None = None,
                                 bins: int = 24):
    if metric not in _DISTRIBUTION_METRICS:
        raise HTTPException(422, f"metric must be one of {sorted(_DISTRIBUTION_METRICS)}")
    if not 1 <= bins <= 200:
        raise HTTPException(422, "bins must be between 1 and 200")
    conn = request.app.state.conn
    clauses, args = [f"{metric} IS NOT NULL"], []
    if model is not None:
        clauses.append("model = ?")
        args.append(model)
    where = f"WHERE {' AND '.join(clauses)}"
    rows = conn.execute(f"SELECT model, {metric} AS v FROM requests {where}", args).fetchall()
    by_model: dict[str | None, list[float]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row["v"])
    groups = []
    for m, values in by_model.items():
        values.sort()
        groups.append({"model": m, "n": len(values), **_quantiles(values),
                       "histogram": _histogram(values, bins)})
    groups.sort(key=lambda g: -g["n"])
    return {"metric": metric, "groups": groups}


@router.get("/models")
async def models_catalog(request: Request):
    conn = request.app.state.conn
    baseline_rows = conn.execute(
        "SELECT b.* FROM model_baselines b INNER JOIN "
        "(SELECT model, MAX(id) AS max_id FROM model_baselines GROUP BY model) latest "
        "ON b.model = latest.model AND b.id = latest.max_id").fetchall()
    agg_rows = conn.execute(
        "SELECT model, COUNT(*) AS n_runs, AVG(tokens_per_second) AS avg_tps, "
        "AVG(time_to_first_token_ms) AS avg_ttft_ms, "
        "SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS error_rate, "
        "MAX(started_at) AS last_used FROM requests WHERE model IS NOT NULL "
        "GROUP BY model").fetchall()
    agg_by_model = {r["model"]: r for r in agg_rows}
    out, seen = [], set()
    for row in baseline_rows:
        seen.add(row["model"])
        out.append(_model_summary(row["model"], row, agg_by_model.get(row["model"])))
    for model, agg in agg_by_model.items():
        if model not in seen:
            out.append(_model_summary(model, None, agg))
    out.sort(key=lambda m: -(m["n_runs"] or 0))
    return {"models": out}


@router.get("/models/detail")
async def model_detail(request: Request, model: str):
    from . import db
    conn = request.app.state.conn
    base = db.get_baseline(conn, model)
    agg = conn.execute(
        "SELECT COUNT(*) AS n_runs, AVG(tokens_per_second) AS avg_tps, "
        "AVG(time_to_first_token_ms) AS avg_ttft_ms, AVG(eval_count) AS avg_eval_count, "
        "SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS error_rate "
        "FROM requests WHERE model = ?", (model,)).fetchone()
    if base is None and not (agg["n_runs"] or 0):
        raise HTTPException(404, "model not found")
    recent = conn.execute(
        "SELECT id, started_at, condition, origin, status_code, error, eval_count, "
        "tokens_per_second, time_to_first_token_ms FROM requests WHERE model = ? "
        "ORDER BY started_at DESC LIMIT 20", (model,)).fetchall()
    evals = _evaluations_for(conn, "r.model = ?", [model])
    return {
        "model": model,
        "digest": base["digest"] if base else None,
        "captured_at": base["captured_at"] if base else None,
        "modelfile_text": base["modelfile_text"] if base else None,
        "parameters_raw": base["parameters_raw"] if base else None,
        "parameters_parsed": _loads_or_empty(base["parameters_parsed_json"]) if base else {},
        "template_text": base["template_text"] if base else None,
        "system_text": base["system_text"] if base else None,
        "details": _loads_or_empty(base["details_json"]) if base else {},
        "n_runs": agg["n_runs"] or 0,
        "avg_tps": _round_or_none(agg["avg_tps"]),
        "avg_ttft_ms": _round_or_none(agg["avg_ttft_ms"]),
        "avg_eval_count": _round_or_none(agg["avg_eval_count"]),
        "error_rate": _round_or_none(agg["error_rate"], 4),
        "recent_requests": [dict(r) for r in recent],
        "evaluations": evals[0] if evals else None,
    }


@router.get("/environments")
async def environments(request: Request):
    conn = request.app.state.conn
    rows = conn.execute(
        "SELECT e.*, COUNT(r.id) AS n_requests, MIN(r.started_at) AS first_seen, "
        "MAX(r.started_at) AS last_seen FROM environments e "
        "LEFT JOIN requests r ON r.environment_id = e.id "
        "GROUP BY e.id ORDER BY e.captured_at DESC").fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["gpu"] = _loads_or_empty(d.pop("gpu_json", None), default=[])
        out.append(d)
    return {"environments": out}


@router.get("/evaluations/summary")
async def evaluations_summary(request: Request, model: str | None = None,
                              experiment: str | None = None):
    conn = request.app.state.conn
    clauses, args = [], []
    for column, value in [("r.model", model), ("r.experiment", experiment)]:
        if value is not None:
            clauses.append(f"{column} = ?")
            args.append(value)
    where = " AND ".join(clauses)
    return {"groups": _evaluations_for(conn, where, args, group_by_condition=True)}


@router.get("/runtime/vram")
async def runtime_vram(request: Request, limit: int = 200):
    conn = request.app.state.conn
    rows = conn.execute(
        "SELECT observed_runtime_json, started_at FROM requests "
        "WHERE observed_runtime_json IS NOT NULL ORDER BY started_at DESC LIMIT ?",
        (limit,)).fetchall()
    points, seen = [], set()
    for row in rows:
        try:
            payload = json.loads(row["observed_runtime_json"])
        except (ValueError, TypeError):
            continue
        observed_at = payload.get("observed_at") or row["started_at"]
        if observed_at in seen:
            continue
        seen.add(observed_at)
        api_ps = payload.get("api_ps") if isinstance(payload, dict) else None
        loaded = api_ps.get("models") if isinstance(api_ps, dict) else None
        models = [{"name": m.get("name") or m.get("model"), "size_vram": m.get("size_vram"),
                  "size": m.get("size"), "until": m.get("expires_at")}
                 for m in (loaded or []) if isinstance(m, dict)]
        points.append({"observed_at": observed_at, "models": models})
    points.reverse()
    return {"points": points}


def _model_summary(model: str, base, agg) -> dict:
    details = _loads_or_empty(base["details_json"]) if base else {}
    return {
        "model": model,
        "digest": base["digest"] if base else None,
        "captured_at": base["captured_at"] if base else None,
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "n_runs": agg["n_runs"] if agg else 0,
        "avg_tps": _round_or_none(agg["avg_tps"]) if agg else None,
        "avg_ttft_ms": _round_or_none(agg["avg_ttft_ms"]) if agg else None,
        "error_rate": _round_or_none(agg["error_rate"], 4) if agg else None,
        "last_used": agg["last_used"] if agg else None,
    }


def _evaluations_for(conn, where: str, args: list, group_by_condition: bool = False) -> list[dict]:
    """Aggregate heuristic metrics (avg), boolean rates and human label counts
    by model (and optionally by condition). Each `evaluations.value` type
    is aggregated according to what it represents."""
    select_condition = "r.condition, " if group_by_condition else ""
    group_condition = ", r.condition" if group_by_condition else ""
    sql = (f"SELECT r.model, {select_condition}e.metric, e.value, e.source "
           f"FROM evaluations e JOIN requests r ON r.id = e.request_id"
           f"{' WHERE ' + where if where else ''} ORDER BY r.model{group_condition}")
    rows = conn.execute(sql, args).fetchall()
    groups: dict[tuple, dict] = {}
    for row in rows:
        condition = row["condition"] if group_by_condition else None
        key = (row["model"], condition)
        g = groups.setdefault(key, {"model": row["model"], "metrics": {}, "bools": {}, "labels": {}})
        if group_by_condition:
            g["condition"] = condition
        value = _loads_or_raw(row["value"])
        if row["source"] == "human" and isinstance(value, dict) and "label" in value:
            g["labels"][value["label"]] = g["labels"].get(value["label"], 0) + 1
        elif isinstance(value, bool):
            b = g["bools"].setdefault(row["metric"], {"true": 0, "n": 0})
            b["n"] += 1
            b["true"] += int(value)
        elif isinstance(value, (int, float)):
            g["metrics"].setdefault(row["metric"], []).append(value)
    out = []
    for g in groups.values():
        out.append({
            "model": g["model"], **({"condition": g["condition"]} if group_by_condition else {}),
            "metrics": {k: {"avg": _avg(v), "n": len(v)} for k, v in g["metrics"].items()},
            "rates": {k: round(b["true"] / b["n"], 4) for k, b in g["bools"].items() if b["n"]},
            "labels": g["labels"],
        })
    out.sort(key=lambda g: (g["model"] or "", g.get("condition") or ""))
    return out


def _quantile(sorted_values: list[float], q: float) -> float:
    idx = q * (len(sorted_values) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def _quantiles(sorted_values: list[float]) -> dict:
    return {"min": sorted_values[0], "p25": _quantile(sorted_values, .25),
            "p50": _quantile(sorted_values, .5), "p75": _quantile(sorted_values, .75),
            "p95": _quantile(sorted_values, .95), "max": sorted_values[-1]}


def _histogram(sorted_values: list[float], bins: int) -> dict:
    lo, hi = sorted_values[0], sorted_values[-1]
    if lo == hi:
        hi = lo + 1.0
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in sorted_values:
        counts[min(bins - 1, int((v - lo) / width))] += 1
    return {"edges": edges, "counts": counts}


def _ns_to_ms(value) -> float | None:
    return None if value is None else round(value / 1e6, 2)


def _round_or_none(value, digits: int = 2):
    return None if value is None else round(value, digits)


def _loads_or_empty(value, default=None):
    if not value:
        return default if default is not None else {}
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default if default is not None else {}


def _loads_or_raw(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _avg(values: list) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
