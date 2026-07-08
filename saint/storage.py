from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

SCHEMA_VERSION = 3


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
    johnny_seat: str | None = None  # resolved seat when the backend was johnny-bound
    state_at_dispatch: str | None = None  # johnny_ready|static_baseline|while_loading|fallback|None
    cache_read_tokens: int | None = None   # provider prompt-cache reads (see migration 0003)
    cache_write_tokens: int | None = None  # provider prompt-cache writes


def build_log_row(decision, *, model_field, backend_latency_ms, success, error_kind,
                  tokens_in, tokens_out, prompt_storage_mode,
                  johnny_seat=None, state_at_dispatch=None,
                  cache_read_tokens=None, cache_write_tokens=None) -> LogRow:
    """Map a RoutingDecision (+ dispatch outcome) to a LogRow. Dispatch-less callers
    (e.g. `saint explain`) pass backend_latency_ms/tokens as None."""
    out = decision.classifier_outcome
    return LogRow(
        request_id=decision.request_id,
        model_field=model_field,
        prefixes_raw=decision.parsed.raw or None,
        pinned_backend=decision.pinned_backend,
        urgency_used=decision.urgency,
        classifier_used=out.classifier_used if out else None,
        classifier_fallback_reason=out.fallback_reason if out else None,
        classifier_input_chars=out.input_chars if out else None,
        classifier_input_truncated_from=out.input_truncated_from if out else None,
        classifier_latency_ms=(out.result.latency_ms if out and out.result else None),
        classifier_domain=(out.result.domain if out and out.result else None),
        classifier_complexity=(out.result.complexity if out and out.result else None),
        classifier_reason=(out.result.reason if out and out.result else None),
        backend_chosen=decision.backend,
        backend_latency_ms=backend_latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        success=success,
        error_kind=error_kind,
        prompt_content=decision.last_user_content_original,
        prompt_storage_mode=prompt_storage_mode,
        johnny_seat=johnny_seat,
        state_at_dispatch=state_at_dispatch,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _load_migration(name: str) -> str:
    return resources.files("saint.migrations").joinpath(name).read_text()


def open_db(path: Path) -> sqlite3.Connection:
    """Open or create the SQLite database. Enables WAL, runs migrations to current version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # autocommit (isolation_level=None); allow cross-thread access for FastAPI workers
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
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
    if current < 2:
        conn.executescript(_load_migration("0002_johnny.sql"))
        conn.execute("PRAGMA user_version = 2")
    if current < 3:
        conn.executescript(_load_migration("0003_cache_metrics.sql"))
        conn.execute("PRAGMA user_version = 3")


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
            tokens_in, tokens_out, success, error_kind, prompt_content, prompt_storage_mode,
            johnny_seat, state_at_dispatch, cache_read_tokens, cache_write_tokens
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
        """,
        (
            datetime.now(UTC).isoformat(),
            row.request_id, row.model_field, row.prefixes_raw, row.pinned_backend,
            row.urgency_used, row.classifier_used, row.classifier_fallback_reason,
            row.classifier_input_chars, row.classifier_input_truncated_from,
            row.classifier_latency_ms, row.classifier_domain, row.classifier_complexity,
            row.classifier_reason, row.backend_chosen, row.backend_latency_ms,
            row.tokens_in, row.tokens_out, 1 if row.success else 0, row.error_kind,
            stored, row.prompt_storage_mode,
            row.johnny_seat, row.state_at_dispatch,
            row.cache_read_tokens, row.cache_write_tokens,
        ),
    )
    return int(cursor.lastrowid or 0)


def fetch_training_rows(conn: sqlite3.Connection, limit: int) -> list[tuple]:
    """Recent (prompt, domain, complexity) rows usable for classifier distillation.

    Excludes rows labeled by the embedding head itself (no self-distillation feedback
    loop) and by the routing caches ('cache'/'inherited' — reused labels, not fresh
    classifier judgments)."""
    return conn.execute(
        "SELECT prompt_content, classifier_domain, classifier_complexity FROM requests "
        "WHERE prompt_content IS NOT NULL AND classifier_domain IS NOT NULL "
        "AND classifier_complexity IS NOT NULL "
        "AND (classifier_used IS NULL OR ("
        "  classifier_used NOT LIKE '%embed-head%' "
        "  AND classifier_used NOT IN ('cache', 'inherited')"
        ")) "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def classifier_traffic_mix(conn: sqlite3.Connection, limit: int) -> dict[str, int]:
    """How the last `limit` classified requests were labeled.

    head   — embedding head answered (confident region)
    llm    — LLM classifier answered (head deferral, or mode='llm')
    reused — routing caches ('cache'/'inherited'); downstream of an earlier decision
    """
    rows = conn.execute(
        "SELECT classifier_used FROM ("
        "  SELECT id, classifier_used FROM requests"
        "  WHERE classifier_used IS NOT NULL ORDER BY id DESC LIMIT ?)",
        (limit,),
    ).fetchall()
    mix = {"head": 0, "llm": 0, "reused": 0}
    for (used,) in rows:
        if "embed-head" in used:
            mix["head"] += 1
        elif used in ("cache", "inherited"):
            mix["reused"] += 1
        else:
            mix["llm"] += 1
    return mix


def count_training_rows_since(conn: sqlite3.Connection, ts: str) -> int:
    """Distinct usable training prompts logged after `ts` (ISO-8601 UTC — the head's
    trained_at). 'How much would a retrain add?'"""
    return conn.execute(
        "SELECT COUNT(DISTINCT prompt_content) FROM requests "
        "WHERE prompt_content IS NOT NULL AND classifier_domain IS NOT NULL "
        "AND classifier_complexity IS NOT NULL "
        "AND (classifier_used IS NULL OR ("
        "  classifier_used NOT LIKE '%embed-head%' "
        "  AND classifier_used NOT IN ('cache', 'inherited')"
        ")) AND ts > ?",
        (ts,),
    ).fetchone()[0]


def clear_requests(conn: sqlite3.Connection) -> int:
    """Delete every request log row and restart ids at 1. Returns rows deleted.

    Truncates through the live connection (schema/WAL intact), so it's safe while
    `saint serve` holds the DB open — unlike deleting the file, which would leave the
    server logging into an unlinked inode.
    """
    (count,) = conn.execute("SELECT COUNT(*) FROM requests").fetchone()
    conn.execute("DELETE FROM requests")
    try:
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'requests'")
    except sqlite3.OperationalError:
        pass  # sqlite_sequence doesn't exist until the first AUTOINCREMENT insert
    conn.execute("VACUUM")
    return int(count)


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
        (backend, datetime.now(UTC).isoformat(), note, row_id),
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
