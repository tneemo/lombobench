"""Endpoints do Lab: sessões de experimentação com painéis lado a lado.

O Lab NÃO tem caminho próprio de captura: o front chama o próprio proxy
(POST /api/chat) com os headers X-Lombo-* preenchidos, e tudo cai nas mesmas
tabelas de qualquer cliente externo.

Modos de sessão:
- controlled (padrão): a cada rodada todos os painéis recebem o MESMO
  histórico-base e só a variável do painel difere — rodadas independentes e
  comparáveis (uma variável por vez).
- branched: após a primeira resposta cada painel segue seu próprio histórico
  (branch_id/parent_request_id registram a árvore).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import db

router = APIRouter()


class LabSession(BaseModel):
    title: str = "Sessão sem título"
    mode: str = Field(default="controlled", pattern="^(controlled|branched)$")
    base_config: dict = Field(default_factory=dict)
    panes: list[dict] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/lab/sessions")
async def create_session(request: Request, body: LabSession):
    conn = request.app.state.conn
    session_id = uuid.uuid4().hex[:12]
    created_at = _now_iso()

    def write(conn):
        conn.execute(
            "INSERT INTO lab_sessions (id, created_at, title, mode, base_config_json, panes_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, created_at, body.title, body.mode,
             json.dumps(body.base_config, ensure_ascii=False),
             json.dumps(body.panes, ensure_ascii=False)))
        conn.commit()

    request.app.state.writer.enqueue(write)
    return {"id": session_id, "created_at": created_at, "title": body.title,
            "mode": body.mode, "base_config": body.base_config, "panes": body.panes}


@router.get("/lab/sessions")
async def list_sessions(request: Request):
    rows = request.app.state.conn.execute(
        "SELECT * FROM lab_sessions ORDER BY created_at DESC").fetchall()
    return {"sessions": [_session_dict(r) for r in rows]}


@router.get("/lab/sessions/{session_id}")
async def get_session(request: Request, session_id: str):
    conn = request.app.state.conn
    row = conn.execute("SELECT * FROM lab_sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "sessão não encontrada")
    result = _session_dict(row)
    reqs = conn.execute(
        "SELECT id, started_at, completed_at, turn_id, branch_id, parent_request_id, "
        "condition, pair_id, replicate, model, response_text, thinking_text, error, "
        "eval_count, tokens_per_second, time_to_first_token_ms, done_reason, "
        "prompt_or_messages_json "
        "FROM requests WHERE lab_session_id = ? ORDER BY started_at", (session_id,)).fetchall()
    result["requests"] = [dict(r) for r in reqs]
    return result


class SessionUpdate(BaseModel):
    title: str | None = None
    mode: str | None = Field(default=None, pattern="^(controlled|branched)$")
    base_config: dict | None = None
    panes: list[dict] | None = None


@router.patch("/lab/sessions/{session_id}")
async def update_session(request: Request, session_id: str, body: SessionUpdate):
    conn = request.app.state.conn
    row = conn.execute("SELECT * FROM lab_sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "sessão não encontrada")
    sets, args = [], []
    for column, value in [("title", body.title), ("mode", body.mode)]:
        if value is not None:
            sets.append(f"{column} = ?")
            args.append(value)
    for column, value in [("base_config_json", body.base_config), ("panes_json", body.panes)]:
        if value is not None:
            sets.append(f"{column} = ?")
            args.append(json.dumps(value, ensure_ascii=False))
    if sets:
        args.append(session_id)
        sql = f"UPDATE lab_sessions SET {', '.join(sets)} WHERE id = ?"

        def write(conn):
            conn.execute(sql, args)
            conn.commit()

        request.app.state.writer.enqueue(write)
    return {"ok": True}


@router.get("/lab/models")
async def list_models(request: Request):
    """Modelos disponíveis no Ollama (para o seletor de painel)."""
    resp = await request.app.state.client.get("/api/tags", timeout=10)
    return resp.json()


def _session_dict(row) -> dict:
    return {"id": row["id"], "created_at": row["created_at"], "title": row["title"],
            "mode": row["mode"],
            "base_config": json.loads(row["base_config_json"] or "{}"),
            "panes": json.loads(row["panes_json"] or "[]")}
