from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LogRow:
    request_id: str
    model_field: str
    prefixes_raw: str | None
    pinned_backend: str | None
    urgency_used: str
    classifier_used: str | None
    classifier_fallback_reason: str | None
    classifier_input_chars: int | None
    classifier_input_truncated_from: int | None
    classifier_latency_ms: int | None
    classifier_domain: str | None
    classifier_complexity: str | None
    classifier_reason: str | None
    backend_chosen: str
    backend_latency_ms: int | None
    tokens_in: int | None
    tokens_out: int | None
    success: bool
    error_kind: str | None
    prompt_content: str | None
    prompt_storage_mode: str  # "full" | "hashed" | "none"


def _load_migration(name: str) -> str:
    return resources.files("goorouter.migrations").joinpath(name).read_text()


def open_db(path: Path) -> sqlite3.Connection:
    """Open or create the SQLite database. Enables WAL, runs migrations to current version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit; we manage txns explicitly
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _migrate(conn)
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _migrate(conn: sqlite3.Connection) -> None:
    current = schema_version(conn)
    if current < 1:
        conn.executescript(_load_migration("0001_init.sql"))
        conn.execute("PRAGMA user_version = 1")


def _apply_storage_mode(content: str | None, mode: str) -> str | None:
    if mode == "full":
        return content
    if mode == "hashed":
        return hashlib.sha256((content or "").encode("utf-8")).hexdigest()
    if mode == "none":
        return None
    raise ValueError(f"unknown prompt_storage_mode: {mode}")


def log_request(conn: sqlite3.Connection, row: LogRow) -> int:
    """Insert a request log row. Returns the new id."""
    stored = _apply_storage_mode(row.prompt_content, row.prompt_storage_mode)
    cursor = conn.execute(
        """
        INSERT INTO requests (
            ts, request_id, model_field, prefixes_raw, pinned_backend, urgency_used,
            classifier_used, classifier_fallback_reason, classifier_input_chars,
            classifier_input_truncated_from, classifier_latency_ms, classifier_domain,
            classifier_complexity, classifier_reason, backend_chosen, backend_latency_ms,
            tokens_in, tokens_out, success, error_kind, prompt_content, prompt_storage_mode
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            row.request_id, row.model_field, row.prefixes_raw, row.pinned_backend,
            row.urgency_used, row.classifier_used, row.classifier_fallback_reason,
            row.classifier_input_chars, row.classifier_input_truncated_from,
            row.classifier_latency_ms, row.classifier_domain, row.classifier_complexity,
            row.classifier_reason, row.backend_chosen, row.backend_latency_ms,
            row.tokens_in, row.tokens_out, 1 if row.success else 0, row.error_kind,
            stored, row.prompt_storage_mode,
        ),
    )
    return int(cursor.lastrowid or 0)


class RelabelError(ValueError):
    pass


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple) -> dict:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


def get_recent(conn: sqlite3.Connection, limit: int = 20, backend: str | None = None) -> list[dict]:
    if backend is None:
        cursor = conn.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,))
    else:
        cursor = conn.execute(
            "SELECT * FROM requests WHERE backend_chosen = ? ORDER BY id DESC LIMIT ?",
            (backend, limit),
        )
    return [_row_to_dict(cursor, r) for r in cursor.fetchall()]


def get_by_id(conn: sqlite3.Connection, row_id: int) -> dict | None:
    cursor = conn.execute("SELECT * FROM requests WHERE id = ?", (row_id,))
    r = cursor.fetchone()
    return _row_to_dict(cursor, r) if r else None


def _set_relabel(conn: sqlite3.Connection, row_id: int, backend: str, note: str | None) -> None:
    conn.execute(
        "UPDATE requests SET relabel_backend = ?, relabel_ts = ?, relabel_note = ? WHERE id = ?",
        (backend, datetime.now(timezone.utc).isoformat(), note, row_id),
    )


def relabel_last(conn: sqlite3.Connection, backend: str, note: str | None) -> int:
    cursor = conn.execute("SELECT MAX(id) FROM requests")
    last_id = cursor.fetchone()[0]
    if last_id is None:
        raise RelabelError("no rows in requests table to relabel")
    _set_relabel(conn, last_id, backend, note)
    return int(last_id)


def relabel_by_id(conn: sqlite3.Connection, row_id: int, backend: str, note: str | None) -> None:
    if get_by_id(conn, row_id) is None:
        raise RelabelError(f"no row with id {row_id}")
    _set_relabel(conn, row_id, backend, note)
