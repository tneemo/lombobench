"""Lab session persistence contracts, using handlers and in-memory SQLite."""

import asyncio
from types import SimpleNamespace

import pytest

from lombo import db
from lombo.lab import LabSession, create_session, get_session


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    yield connection
    connection.close()


def _create_and_read_session(conn, body):
    writes = []
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        conn=conn, writer=SimpleNamespace(enqueue=writes.append),
    )))
    result = asyncio.run(create_session(request, body))
    for write in writes:
        write(conn)
    stored = conn.execute(
        "SELECT title FROM lab_sessions WHERE id = ?", (result["id"],),
    ).fetchone()
    retrieved = asyncio.run(get_session(request, result["id"]))
    return result, stored, retrieved


def test_create_session_without_title_stores_english_default(conn):
    result, stored, retrieved = _create_and_read_session(conn, LabSession())

    assert result["title"] == "Untitled session"
    assert stored["title"] == "Untitled session"
    assert retrieved["title"] == "Untitled session"


@pytest.mark.parametrize("title", ["My experiment", "Sessão personalizada", "Sessão sem título"])
def test_create_session_preserves_explicit_title(conn, title):
    result, stored, retrieved = _create_and_read_session(conn, LabSession(title=title))

    assert result["title"] == title
    assert stored["title"] == title
    assert retrieved["title"] == title
