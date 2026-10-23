"""Aggregation endpoint contracts (/_lombo/api/metrics|models|environments|...).

Seed an in-memory SQLite database with the real db.py helpers (the same write
paths used in production) and call router functions directly, following
the style of test_diagnostics_endpoint.py.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from lombo import db
from lombo.insights import (
    environments,
    evaluations_summary,
    metrics_distributions,
    metrics_latency,
    metrics_timeseries,
    model_detail,
    models_catalog,
    runtime_vram,
)
from lombo.lab import session_tree


def _request(conn):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(conn=conn)))


def _insert_request(conn, **overrides):
    record = {
        "id": overrides.pop("id", "req-" + str(id(overrides))),
        "started_at": "2026-07-10T10:00:00+00:00",
        "endpoint": "/api/chat",
        "model": "llama3:8b",
        "origin": "external",
        "status_code": 200,
        "tokens_per_second": 40.0,
        "time_to_first_token_ms": 200.0,
        "eval_count": 50,
        "load_duration_ns": 500_000_000,
        "prompt_eval_duration_ns": 100_000_000,
        "eval_duration_ns": 1_000_000_000,
        "prompt_eval_count": 12,
    }
    record.update(overrides)
    db.insert_request(conn, record)
    return record["id"]


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


# --------------------------- metrics/timeseries -----------------------------

def test_timeseries_groups_by_hour_and_filters_model(conn):
    _insert_request(conn, id="a", started_at="2026-07-10T10:05:00+00:00", model="llama3:8b")
    _insert_request(conn, id="b", started_at="2026-07-10T10:40:00+00:00", model="llama3:8b")
    _insert_request(conn, id="c", started_at="2026-07-10T11:05:00+00:00", model="other:1b")

    result = asyncio.run(metrics_timeseries(_request(conn), model="llama3:8b",
                                            experiment=None, origin=None, bucket="hour"))

    assert result["bucket"] == "hour"
    assert len(result["points"]) == 1
    assert result["points"][0]["n"] == 2
    assert result["points"][0]["t"] == "2026-07-10T10"


def test_timeseries_rejects_invalid_bucket(conn):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(metrics_timeseries(_request(conn), model=None, experiment=None,
                                       origin=None, bucket="minute"))
    assert exc.value.status_code == 422


def test_timeseries_counts_errors(conn):
    _insert_request(conn, id="ok", status_code=200, error=None)
    _insert_request(conn, id="bad", status_code=500, error="boom")
    result = asyncio.run(metrics_timeseries(_request(conn), model=None, experiment=None,
                                            origin=None, bucket="day"))
    assert result["points"][0]["errors"] == 1
    assert result["points"][0]["n"] == 2


# ----------------------------- metrics/latency -------------------------------

def test_latency_converts_ns_to_ms_and_groups_by_model(conn):
    _insert_request(conn, id="a", model="llama3:8b", load_duration_ns=2_000_000_000)
    _insert_request(conn, id="b", model="llama3:8b", load_duration_ns=1_000_000_000)

    result = asyncio.run(metrics_latency(_request(conn), model=None, experiment=None, limit=500))

    assert result["groups"][0]["model"] == "llama3:8b"
    assert result["groups"][0]["avg_load_ms"] == 1500.0
    assert len(result["points"]) == 2
    assert result["points"][0]["load_ms"] in (1000.0, 2000.0)


def test_latency_excludes_non_200(conn):
    _insert_request(conn, id="a", status_code=200)
    _insert_request(conn, id="b", status_code=500, error="boom")
    result = asyncio.run(metrics_latency(_request(conn), model=None, experiment=None, limit=500))
    assert sum(g["n"] for g in result["groups"]) == 1


# -------------------------- metrics/distributions ----------------------------

def test_distributions_computes_quantiles_and_histogram(conn):
    for i, v in enumerate([10.0, 20.0, 30.0, 40.0, 50.0]):
        _insert_request(conn, id=f"r{i}", tokens_per_second=v)

    result = asyncio.run(metrics_distributions(_request(conn), metric="tokens_per_second",
                                               model=None, bins=5))

    group = result["groups"][0]
    assert group["n"] == 5
    assert group["min"] == 10.0
    assert group["max"] == 50.0
    assert group["p50"] == 30.0
    assert sum(group["histogram"]["counts"]) == 5
    assert len(group["histogram"]["edges"]) == 6


def test_distributions_rejects_unknown_metric(conn):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(metrics_distributions(_request(conn), metric="response_text",
                                          model=None, bins=10))
    assert exc.value.status_code == 422


def test_distributions_rejects_bins_out_of_range(conn):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(metrics_distributions(_request(conn), metric="tokens_per_second",
                                          model=None, bins=0))
    assert exc.value.status_code == 422


# --------------------------------- models ------------------------------------

def test_models_catalog_merges_baseline_and_aggregates(conn):
    db.insert_baseline(conn, {
        "model": "llama3:8b", "digest": "sha256:abc", "captured_at": "2026-07-01T00:00:00+00:00",
        "details_json": json.dumps({"family": "llama", "parameter_size": "8B",
                                    "quantization_level": "Q4_K_M"}),
    })
    _insert_request(conn, id="a", model="llama3:8b", tokens_per_second=44.0)
    _insert_request(conn, id="b", model="llama3:8b", tokens_per_second=36.0)
    _insert_request(conn, id="c", model="untracked/model", tokens_per_second=90.0)

    result = asyncio.run(models_catalog(_request(conn)))

    by_model = {m["model"]: m for m in result["models"]}
    assert by_model["llama3:8b"]["family"] == "llama"
    assert by_model["llama3:8b"]["quantization_level"] == "Q4_K_M"
    assert by_model["llama3:8b"]["n_runs"] == 2
    assert by_model["llama3:8b"]["avg_tps"] == 40.0
    # a model without a baseline still appears with empty fields; it is not discarded
    assert by_model["untracked/model"]["family"] is None
    assert by_model["untracked/model"]["n_runs"] == 1


def test_models_catalog_picks_most_recent_baseline(conn):
    db.insert_baseline(conn, {"model": "m", "digest": "old",
                              "captured_at": "2026-07-01T00:00:00+00:00"})
    db.insert_baseline(conn, {"model": "m", "digest": "new",
                              "captured_at": "2026-07-05T00:00:00+00:00"})
    result = asyncio.run(models_catalog(_request(conn)))
    assert result["models"][0]["digest"] == "new"


def test_model_detail_includes_recent_requests_and_404s_when_absent(conn):
    db.insert_baseline(conn, {"model": "llama3:8b", "digest": "sha256:abc",
                              "captured_at": "2026-07-01T00:00:00+00:00",
                              "parameters_raw": "temperature 0.7"})
    _insert_request(conn, id="a", model="llama3:8b")

    result = asyncio.run(model_detail(_request(conn), model="llama3:8b"))
    assert result["digest"] == "sha256:abc"
    assert result["n_runs"] == 1
    assert len(result["recent_requests"]) == 1

    with pytest.raises(HTTPException) as exc:
        asyncio.run(model_detail(_request(conn), model="nope/model"))
    assert exc.value.status_code == 404


# ------------------------------ environments ----------------------------------

def test_environments_counts_requests_and_parses_gpu(conn):
    env_id = db.get_or_create_environment(conn, {
        "fingerprint": "fp1", "captured_at": "2026-07-01T00:00:00+00:00",
        "os": "Linux", "cpu": "x86", "ram_gb": 32.0,
        "gpu_json": json.dumps([{"name": "RTX 4090", "memory": "24576 MiB"}]),
    })
    _insert_request(conn, id="a", environment_id=env_id)
    _insert_request(conn, id="b", environment_id=env_id)

    result = asyncio.run(environments(_request(conn)))

    assert result["environments"][0]["n_requests"] == 2
    assert result["environments"][0]["gpu"][0]["name"] == "RTX 4090"


def test_environments_empty_gpu_json_becomes_empty_list(conn):
    db.get_or_create_environment(conn, {
        "fingerprint": "fp2", "captured_at": "2026-07-01T00:00:00+00:00",
        "gpu_json": None,
    })
    result = asyncio.run(environments(_request(conn)))
    assert result["environments"][0]["gpu"] == []


# --------------------------- evaluations/summary -------------------------------

def test_evaluations_summary_aggregates_numeric_bool_and_labels(conn):
    rid = _insert_request(conn, id="a", model="llama3:8b", condition="base")
    db.insert_evaluations(conn, rid, [
        ("repetition_rate", 0.1), ("truncated", True), ("json_valid", True),
    ], "heuristic", "2026-07-10T10:00:00+00:00")
    rid2 = _insert_request(conn, id="b", model="llama3:8b", condition="base")
    db.insert_evaluations(conn, rid2, [
        ("repetition_rate", 0.3), ("truncated", False),
    ], "heuristic", "2026-07-10T10:01:00+00:00")
    db.insert_evaluations(conn, rid2, [
        ("label", {"label": "correto", "note": None}),
    ], "human", "2026-07-10T10:02:00+00:00")

    result = asyncio.run(evaluations_summary(_request(conn), model=None, experiment=None))

    group = result["groups"][0]
    assert group["model"] == "llama3:8b"
    assert group["condition"] == "base"
    assert group["metrics"]["repetition_rate"]["avg"] == 0.2
    assert group["metrics"]["repetition_rate"]["n"] == 2
    assert group["rates"]["truncated"] == 0.5
    assert group["rates"]["json_valid"] == 1.0
    assert group["labels"]["correto"] == 1


def test_evaluations_summary_filters_by_model(conn):
    rid = _insert_request(conn, id="a", model="A")
    db.insert_evaluations(conn, rid, [("repetition_rate", 0.5)], "heuristic", "t")
    rid2 = _insert_request(conn, id="b", model="B")
    db.insert_evaluations(conn, rid2, [("repetition_rate", 0.9)], "heuristic", "t")

    result = asyncio.run(evaluations_summary(_request(conn), model="A", experiment=None))
    assert len(result["groups"]) == 1
    assert result["groups"][0]["model"] == "A"


# ------------------------------- runtime/vram -----------------------------------

def test_runtime_vram_parses_api_ps_and_dedupes(conn):
    payload = json.dumps({
        "api_ps": {"models": [{"name": "llama3:8b", "size_vram": 5_000_000_000,
                               "size": 6_000_000_000, "expires_at": "2026-07-10T12:00:00Z"}]},
        "observed_at": "2026-07-10T10:00:00+00:00",
    })
    _insert_request(conn, id="a", observed_runtime_json=payload)
    _insert_request(conn, id="b", observed_runtime_json=payload)  # same observed_at

    result = asyncio.run(runtime_vram(_request(conn), limit=200))

    assert len(result["points"]) == 1  # deduplicate by observed_at
    assert result["points"][0]["models"][0]["name"] == "llama3:8b"
    assert result["points"][0]["models"][0]["size_vram"] == 5_000_000_000


def test_runtime_vram_tolerates_malformed_json(conn):
    _insert_request(conn, id="a", observed_runtime_json="{not valid json")
    _insert_request(conn, id="b", observed_runtime_json=json.dumps({"api_ps": "unexpected-string"}))
    result = asyncio.run(runtime_vram(_request(conn), limit=200))
    # malformed data is ignored; the second yields a point without models (api_ps is not a dict)
    assert len(result["points"]) == 1
    assert result["points"][0]["models"] == []


# ------------------------------ lab/sessions/tree --------------------------------

def test_session_tree_returns_ordered_nodes(conn):
    db.insert_request(conn, {"id": "s1", "created_at": "2026-07-10T10:00:00+00:00"}) if False else None
    conn.execute(
        "INSERT INTO lab_sessions (id, created_at, title, mode, base_config_json, panes_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("sess-1", "2026-07-10T10:00:00+00:00", "t", "branched", "{}", "[]"))
    conn.commit()
    _insert_request(conn, id="r1", lab_session_id="sess-1", branch_id="pane-0",
                    started_at="2026-07-10T10:00:00+00:00")
    _insert_request(conn, id="r2", lab_session_id="sess-1", branch_id="pane-0",
                    parent_request_id="r1", started_at="2026-07-10T10:01:00+00:00")

    result = asyncio.run(session_tree(_request(conn), "sess-1"))

    assert result["session_id"] == "sess-1"
    assert [n["id"] for n in result["nodes"]] == ["r1", "r2"]
    assert result["nodes"][1]["parent_request_id"] == "r1"


def test_session_tree_404_when_missing(conn):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(session_tree(_request(conn), "nope"))
    assert exc.value.status_code == 404
