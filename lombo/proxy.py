"""LomboProxy — proxy transparente para o Ollama com captura em SQLite.

Streaming transparente com baixo overhead mensurável: os chunks fluem ao
cliente à medida que chegam do Ollama e são acumulados em paralelo para o
registro. O overhead real é medido pelo script de verificação (direto ×
intermediado) — nunca alegado como zero.

Privacidade: o servidor escuta 127.0.0.1 por padrão; `X-Lombo-No-Log` faz a
requisição passar sem registro; `--no-capture` desliga a captura global.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import baseline as bl
from . import behavior, capture, db, environment

log = logging.getLogger("lombo.proxy")

STATIC_DIR = Path(__file__).parent / "static"

# Headers que não devem ser repassados tal-e-qual (hop-by-hop / recalculados).
_SKIP_REQUEST_HEADERS = {"host", "content-length", "connection", "keep-alive",
                         "transfer-encoding", "upgrade", "proxy-connection"}
_SKIP_RESPONSE_HEADERS = {"content-length", "connection", "keep-alive",
                          "transfer-encoding", "content-encoding"}

# Metadados experimentais aceitos via header (removidos antes do repasse).
_LOMBO_HEADERS = {
    "x-lombo-experiment": "experiment",
    "x-lombo-condition": "condition",
    "x-lombo-pair": "pair_id",
    "x-lombo-replicate": "replicate",
    "x-lombo-session": "lab_session_id",
    "x-lombo-turn": "turn_id",
    "x-lombo-branch": "branch_id",
    "x-lombo-parent": "parent_request_id",
    "x-lombo-origin": "origin",
}


@dataclass
class LomboConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    db_path: str = "lombo.db"
    capture: bool = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(config: LomboConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.conn = db.connect(config.db_path)
        app.state.writer = db.DbWriter(app.state.conn)
        app.state.writer.start()
        app.state.client = httpx.AsyncClient(base_url=config.ollama_url, timeout=None)
        app.state.baseline_cache = {}
        env = await environment.snapshot(config.ollama_url)
        app.state.environment_id = db.get_or_create_environment(app.state.conn, env)
        app.state.ollama_version = env.get("ollama_version")
        yield
        await app.state.client.aclose()
        await app.state.writer.close()
        app.state.conn.close()

    app = FastAPI(lifespan=lifespan)
    app.state.config = config

    if STATIC_DIR.exists():
        app.mount("/_lombo/app", StaticFiles(directory=STATIC_DIR, html=True), name="lab")

    @app.get("/_lombo/health")
    async def health():
        return {"ok": True, "capture": config.capture,
                "ollama_url": config.ollama_url,
                "ollama_version": app.state.ollama_version,
                "notice": "As conversas interceptadas são gravadas localmente e "
                          "podem conter dados sensíveis."}

    from .analysis import router as analysis_router
    from .lab import router as lab_router
    app.include_router(analysis_router, prefix="/_lombo/api")
    app.include_router(lab_router, prefix="/_lombo/api")

    @app.get("/_lombo/")
    async def lab_index():
        from fastapi.responses import FileResponse
        return FileResponse(STATIC_DIR / "index.html")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
    async def proxy(request: Request, path: str):
        return await _forward(app, config, request, "/" + path)

    return app


async def _forward(app: FastAPI, config: LomboConfig, request: Request, endpoint: str):
    client: httpx.AsyncClient = app.state.client
    body = await request.body()

    meta: dict = {}
    no_log = False
    fwd_headers = []
    for name, value in request.headers.items():
        lname = name.lower()
        if lname == "x-lombo-no-log":
            no_log = True
            continue
        if lname in _LOMBO_HEADERS:
            # texto livre (ex.: condition) chega percent-encoded — headers só
            # aceitam ISO-8859-1; unquote() é no-op em valores já ASCII simples.
            meta[_LOMBO_HEADERS[lname]] = unquote(value)
            continue
        if lname in _SKIP_REQUEST_HEADERS or lname.startswith("x-lombo-"):
            continue
        fwd_headers.append((name, value))

    should_capture = (
        config.capture and not no_log
        and request.method == "POST"
        and endpoint in capture.CAPTURED_ENDPOINTS
    )

    upstream_req = client.build_request(request.method, endpoint,
                                        headers=fwd_headers, content=body)
    started_at = _now_iso()
    started_perf = time.perf_counter()
    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as exc:
        error = f"ollama inacessível: {exc}"
        if should_capture:
            _finalize_error(app, endpoint, body, meta, started_at, error)
        return JSONResponse({"error": error}, status_code=502)

    resp_headers = {k: v for k, v in upstream.headers.items()
                    if k.lower() not in _SKIP_RESPONSE_HEADERS}

    if not should_capture:
        async def passthrough():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
        return StreamingResponse(passthrough(), status_code=upstream.status_code,
                                 headers=resp_headers)

    try:
        req_body = json.loads(body)
    except ValueError:
        req_body = {}
    req_info = capture.parse_request(endpoint, req_body)
    acc = capture.StreamAccumulator(endpoint, req_info["stream"])
    timing = {"first_token_perf": None}

    async def tee():
        error = None
        try:
            async for chunk in upstream.aiter_raw():
                if chunk and timing["first_token_perf"] is None:
                    timing["first_token_perf"] = time.perf_counter()
                    timing["first_token_at"] = _now_iso()
                acc.feed(chunk)
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            error = "client_disconnected"
            raise
        except httpx.HTTPError as exc:
            error = f"stream interrompido: {exc}"
        finally:
            await upstream.aclose()
            completed_at = _now_iso()
            asyncio.get_running_loop().create_task(_finalize(
                app, endpoint, body, req_body, req_info, meta, acc,
                started_at, started_perf, timing, completed_at,
                upstream.status_code, error,
            ))

    return StreamingResponse(tee(), status_code=upstream.status_code, headers=resp_headers)


def _finalize_error(app, endpoint, body, meta, started_at, error):
    record = {
        "id": uuid.uuid4().hex, "started_at": started_at, "completed_at": _now_iso(),
        "endpoint": endpoint, "stream": 0, "origin": meta.get("origin", "external"),
        "request_body_raw": body.decode("utf-8", "replace"),
        "status_code": 502, "error": error,
        "environment_id": app.state.environment_id,
        **{k: meta.get(k) for k in ("experiment", "condition", "pair_id",
                                    "lab_session_id", "turn_id", "branch_id",
                                    "parent_request_id")},
    }
    if meta.get("replicate"):
        record["replicate"] = _int_or_none(meta["replicate"])
    try:
        record["model"] = json.loads(body).get("model")
    except ValueError:
        pass
    app.state.writer.enqueue(lambda conn: db.insert_request(conn, record))


async def _finalize(app, endpoint, body, req_body, req_info, meta, acc,
                    started_at, started_perf, timing, completed_at,
                    status_code, error):
    """Monta o registro completo e o enfileira para o escritor único.

    Falha aqui vira warning — jamais afeta a resposta já entregue ao cliente.
    """
    try:
        result = acc.result()
        model = req_info.get("model")
        modelfile_params = await _ensure_baseline(app, model) if model else {}
        inferred, diff, _ = bl.layer_params(
            req_info["requested_params_ollama"], modelfile_params)
        confirmed = bl.confirmed_from_response(result["final_chunk"])
        observed = await _observed_runtime(app)

        metrics = result["metrics"]
        tps = capture.tokens_per_second(metrics)
        ttft_ms = None
        if timing.get("first_token_perf") is not None:
            ttft_ms = (timing["first_token_perf"] - started_perf) * 1000.0

        record = {
            "id": uuid.uuid4().hex,
            "started_at": started_at,
            "first_token_at": timing.get("first_token_at"),
            "completed_at": completed_at,
            "time_to_first_token_ms": ttft_ms,
            "endpoint": endpoint,
            "model": model,
            "stream": 1 if req_info["stream"] else 0,
            "origin": meta.get("origin", "external"),
            "lab_session_id": meta.get("lab_session_id"),
            "turn_id": meta.get("turn_id"),
            "branch_id": meta.get("branch_id"),
            "parent_request_id": meta.get("parent_request_id"),
            "experiment": meta.get("experiment"),
            "condition": meta.get("condition"),
            "pair_id": meta.get("pair_id"),
            "replicate": _int_or_none(meta.get("replicate")),
            "environment_id": app.state.environment_id,
            "request_body_raw": body.decode("utf-8", "replace"),
            "requested_params_json": _dumps(req_info["requested_params"]),
            "inferred_params_json": _dumps(inferred),
            "confirmed_params_json": _dumps(confirmed),
            "observed_runtime_json": _dumps(observed) if observed else None,
            "params_diff_json": _dumps(diff),
            "system_prompt": req_info.get("system_prompt"),
            "prompt_or_messages_json": _dumps(req_info.get("prompt_or_messages")),
            "response_text": result["response_text"],
            "thinking_text": result["thinking_text"],
            "status_code": status_code,
            "error": error or result.get("parse_error"),
            "total_duration_ns": metrics.get("total_duration"),
            "load_duration_ns": metrics.get("load_duration"),
            "prompt_eval_count": metrics.get("prompt_eval_count"),
            "prompt_eval_duration_ns": metrics.get("prompt_eval_duration"),
            "eval_count": metrics.get("eval_count"),
            "eval_duration_ns": metrics.get("eval_duration"),
            "tokens_per_second": tps,
            "done_reason": result.get("done_reason"),
        }
        evals = behavior.evaluate(record, req_body) if status_code == 200 else []
        created_at = _now_iso()
        request_id = record["id"]

        def write(conn):
            db.insert_request(conn, record)
            if evals:
                db.insert_evaluations(conn, request_id, evals, "heuristic", created_at)

        app.state.writer.enqueue(write)
    except Exception:
        log.warning("falha ao montar registro de %s (descartado)", endpoint, exc_info=True)


async def _ensure_baseline(app, model: str) -> dict:
    """Garante snapshot do baseline do modelo; retorna os params do Modelfile parseados."""
    cache = app.state.baseline_cache
    if model in cache:
        return cache[model]
    modelfile_params: dict = {}
    try:
        row = db.get_baseline(app.state.conn, model)
        client: httpx.AsyncClient = app.state.client
        resp = await client.post("/api/show", json={"model": model}, timeout=30)
        if resp.status_code == 200:
            show = resp.json()
            digest = await _model_digest(client, model)
            parameters_raw = show.get("parameters")
            modelfile_params = bl.parse_parameters_raw(parameters_raw)
            if row is None or (digest and row["digest"] != digest):
                db_row = {
                    "model": model, "digest": digest, "captured_at": _now_iso(),
                    "modelfile_text": show.get("modelfile"),
                    "parameters_raw": parameters_raw,
                    "parameters_parsed_json": _dumps(modelfile_params),
                    "template_text": show.get("template"),
                    "system_text": show.get("system"),
                    "details_json": _dumps(show.get("details")),
                }
                app.state.writer.enqueue(lambda conn: db.insert_baseline(conn, db_row))
        elif row is not None:
            modelfile_params = json.loads(row["parameters_parsed_json"] or "{}")
    except Exception:
        log.warning("não foi possível obter baseline de %s", model, exc_info=True)
    cache[model] = modelfile_params
    return modelfile_params


async def _model_digest(client: httpx.AsyncClient, model: str) -> str | None:
    try:
        resp = await client.get("/api/tags", timeout=10)
        for m in resp.json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return m.get("digest")
    except Exception:
        pass
    return None


async def _observed_runtime(app) -> dict | None:
    """Estado observado do runtime (/api/ps). NÃO é confirmação de parâmetro
    desta requisição: o modelo carregado pode ser compartilhado ou ter mudado."""
    try:
        client: httpx.AsyncClient = app.state.client
        resp = await client.get("/api/ps", timeout=5)
        if resp.status_code == 200:
            return {"api_ps": resp.json(), "observed_at": _now_iso(),
                    "note": "estado do runtime; não confirma parâmetros da requisição"}
    except Exception:
        pass
    return None


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _int_or_none(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
