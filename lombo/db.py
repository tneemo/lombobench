"""SQLite schema and database access for LomboBench.

All WRITES pass through a queue with a single writer (`DbWriter`): Lab panes
and external clients can finish generations simultaneously, and writes must
never contend for the database in parallel (avoids "database is locked").
Reads are performed directly (WAL allows concurrent readers).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import Any, Callable, Iterable

log = logging.getLogger("lombo.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                      TEXT PRIMARY KEY,
    -- generation timeline (measured by the proxy)
    started_at              TEXT NOT NULL,
    first_token_at          TEXT,
    completed_at            TEXT,
    time_to_first_token_ms  REAL,
    -- identification
    endpoint                TEXT NOT NULL,
    model                   TEXT,
    stream                  INTEGER NOT NULL DEFAULT 0,
    origin                  TEXT NOT NULL DEFAULT 'external',  -- 'lab' | 'external'
    lab_session_id          TEXT,
    turn_id                 TEXT,
    branch_id               TEXT,
    parent_request_id       TEXT,
    -- experimental design
    experiment              TEXT,
    condition               TEXT,
    pair_id                 TEXT,
    replicate               INTEGER,
    environment_id          INTEGER REFERENCES environments(id),
    -- audit/replay: full body, never reconstructed
    request_body_raw        TEXT,
    -- parameter layers: NEVER merged
    requested_params_json   TEXT,   -- exactly what the client sent
    inferred_params_json    TEXT,   -- effective values inferred by the proxy (with provenance)
    confirmed_params_json   TEXT,   -- only what this request's response confirms
    observed_runtime_json   TEXT,   -- runtime state (/api/ps); NOT confirmation
    params_diff_json        TEXT,   -- what differs from the model baseline
    -- content
    system_prompt           TEXT,
    prompt_or_messages_json TEXT,
    response_text           TEXT,
    thinking_text           TEXT,   -- separate from the response, never concatenated
    status_code             INTEGER,
    error                   TEXT,
    -- metrics reported by Ollama
    total_duration_ns       INTEGER,
    load_duration_ns        INTEGER,
    prompt_eval_count       INTEGER,
    prompt_eval_duration_ns INTEGER,
    eval_count              INTEGER,
    eval_duration_ns        INTEGER,
    tokens_per_second       REAL,
    done_reason             TEXT,
    -- Phase 2 (autonomous regulator)
    regulator_action_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_model      ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_pair       ON requests(pair_id);
CREATE INDEX IF NOT EXISTS idx_requests_experiment ON requests(experiment);
CREATE INDEX IF NOT EXISTS idx_requests_session    ON requests(lab_session_id);

CREATE TABLE IF NOT EXISTS model_baselines (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    model                  TEXT NOT NULL,
    digest                 TEXT,
    captured_at            TEXT NOT NULL,
    modelfile_text         TEXT,
    parameters_raw         TEXT,   -- `parameters` block from POST /api/show, original text
    parameters_parsed_json TEXT,   -- derived from raw data; never replaces it
    template_text          TEXT,
    system_text            TEXT,
    details_json           TEXT
);
CREATE INDEX IF NOT EXISTS idx_baselines_model ON model_baselines(model);

CREATE TABLE IF NOT EXISTS environments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at    TEXT NOT NULL,
    fingerprint    TEXT UNIQUE,
    ollama_version TEXT,
    os             TEXT,
    os_version     TEXT,
    python_version TEXT,
    hostname       TEXT,
    cpu            TEXT,
    ram_gb         REAL,
    gpu_json       TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL REFERENCES requests(id),
    metric     TEXT NOT NULL,
    value      TEXT,
    source     TEXT NOT NULL,   -- 'heuristic' | 'human' | 'judge'
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evaluations_request ON evaluations(request_id);

CREATE TABLE IF NOT EXISTS lab_sessions (
    id               TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    title            TEXT,
    mode             TEXT NOT NULL DEFAULT 'controlled',  -- 'controlled' | 'branched'
    base_config_json TEXT,
    panes_json       TEXT
);
"""

REQUEST_COLUMNS = [
    "id", "started_at", "first_token_at", "completed_at", "time_to_first_token_ms",
    "endpoint", "model", "stream", "origin", "lab_session_id", "turn_id",
    "branch_id", "parent_request_id", "experiment", "condition", "pair_id",
    "replicate", "environment_id", "request_body_raw", "requested_params_json",
    "inferred_params_json", "confirmed_params_json", "observed_runtime_json",
    "params_diff_json", "system_prompt", "prompt_or_messages_json",
    "response_text", "thinking_text", "status_code", "error",
    "total_duration_ns", "load_duration_ns", "prompt_eval_count",
    "prompt_eval_duration_ns", "eval_count", "eval_duration_ns",
    "tokens_per_second", "done_reason", "regulator_action_json",
]


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def insert_request(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
    cols = [c for c in REQUEST_COLUMNS if c in record]
    sql = f"INSERT INTO requests ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    conn.execute(sql, [record[c] for c in cols])
    conn.commit()


def insert_evaluations(
    conn: sqlite3.Connection,
    request_id: str,
    items: Iterable[tuple[str, Any]],
    source: str,
    created_at: str,
) -> None:
    conn.executemany(
        "INSERT INTO evaluations (request_id, metric, value, source, created_at) VALUES (?, ?, ?, ?, ?)",
        [(request_id, metric, json.dumps(value, ensure_ascii=False), source, created_at)
         for metric, value in items],
    )
    conn.commit()


def get_baseline(conn: sqlite3.Connection, model: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM model_baselines WHERE model = ? ORDER BY id DESC LIMIT 1", (model,)
    ).fetchone()


def insert_baseline(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = ["model", "digest", "captured_at", "modelfile_text", "parameters_raw",
            "parameters_parsed_json", "template_text", "system_text", "details_json"]
    conn.execute(
        f"INSERT INTO model_baselines ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [row.get(c) for c in cols],
    )
    conn.commit()


def get_or_create_environment(conn: sqlite3.Connection, env: dict[str, Any]) -> int:
    existing = conn.execute(
        "SELECT id FROM environments WHERE fingerprint = ?", (env["fingerprint"],)
    ).fetchone()
    if existing:
        return existing["id"]
    cols = ["captured_at", "fingerprint", "ollama_version", "os", "os_version",
            "python_version", "hostname", "cpu", "ram_gb", "gpu_json"]
    cur = conn.execute(
        f"INSERT INTO environments ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [env.get(c) for c in cols],
    )
    conn.commit()
    return cur.lastrowid


class DbWriter:
    """Asynchronous queue with a single consumer: serializes all writes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._queue: asyncio.Queue[Callable[[sqlite3.Connection], None] | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while True:
            fn = await self._queue.get()
            if fn is None:
                break
            try:
                fn(self._conn)
            except Exception:  # never stop the writer because of one record
                log.exception("failed to write record (record discarded)")
            finally:
                self._queue.task_done()

    def enqueue(self, fn: Callable[[sqlite3.Connection], None]) -> None:
        self._queue.put_nowait(fn)

    async def close(self) -> None:
        if self._task is not None:
            await self._queue.put(None)
            await self._task
            self._task = None
