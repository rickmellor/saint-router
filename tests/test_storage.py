import hashlib
from pathlib import Path

import pytest

from saint.storage import SCHEMA_VERSION, LogRow, log_request, open_db, schema_version


def test_open_db_creates_schema(tmp_path: Path):
    db = tmp_path / "log.sqlite"
    conn = open_db(db)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'")
    assert cursor.fetchone() is not None
    assert schema_version(conn) == SCHEMA_VERSION


def test_open_db_enables_wal(tmp_path: Path):
    db = tmp_path / "log.sqlite"
    conn = open_db(db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_open_db_idempotent(tmp_path: Path):
    db = tmp_path / "log.sqlite"
    open_db(db).close()
    # Second open shouldn't error
    conn = open_db(db)
    assert schema_version(conn) == SCHEMA_VERSION


def _row(**overrides) -> LogRow:
    base = LogRow(
        request_id="r-1",
        model_field="saint-auto",
        prefixes_raw=None,
        pinned_backend=None,
        urgency_used="normal",
        classifier_used="local-small",
        classifier_fallback_reason=None,
        classifier_input_chars=200,
        classifier_input_truncated_from=None,
        classifier_latency_ms=180,
        classifier_domain="code",
        classifier_complexity="medium",
        classifier_reason="standard refactor",
        backend_chosen="local-coder",
        backend_latency_ms=4200,
        tokens_in=120,
        tokens_out=480,
        success=True,
        error_kind=None,
        prompt_content="please refactor this",
        prompt_storage_mode="full",
    )
    return base if not overrides else LogRow(**{**base.__dict__, **overrides})


def test_log_request_full(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    row_id = log_request(conn, _row())
    assert row_id == 1
    r = conn.execute("SELECT prompt_content, prompt_storage_mode, success FROM requests").fetchone()
    assert r == ("please refactor this", "full", 1)


def test_log_request_hashed_storage(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(prompt_storage_mode="hashed"))
    r = conn.execute("SELECT prompt_content, prompt_storage_mode FROM requests").fetchone()
    assert len(r[0]) == 64
    assert r[0] == hashlib.sha256(b"please refactor this").hexdigest()
    assert r[1] == "hashed"


def test_log_request_none_storage(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(prompt_storage_mode="none"))
    r = conn.execute("SELECT prompt_content, prompt_storage_mode FROM requests").fetchone()
    assert r == (None, "none")


from saint.storage import RelabelError, get_by_id, get_recent, relabel_by_id, relabel_last


def test_get_recent_orders_by_ts_desc(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(request_id="r-1"))
    log_request(conn, _row(request_id="r-2"))
    log_request(conn, _row(request_id="r-3"))
    rows = get_recent(conn, limit=2)
    ids = [r["request_id"] for r in rows]
    assert ids == ["r-3", "r-2"]


def test_get_recent_filtered_by_backend(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(backend_chosen="local-coder"))
    log_request(conn, _row(backend_chosen="cloud-large"))
    rows = get_recent(conn, limit=10, backend="local-coder")
    assert len(rows) == 1


def test_get_by_id(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    rid = log_request(conn, _row())
    row = get_by_id(conn, rid)
    assert row is not None
    assert row["id"] == rid


def test_relabel_last(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(request_id="a"))
    log_request(conn, _row(request_id="b"))
    relabel_last(conn, "cloud-large", note="should have been bigger")
    r = conn.execute(
        "SELECT request_id, relabel_backend, relabel_note FROM requests ORDER BY id DESC"
    ).fetchone()
    assert r == ("b", "cloud-large", "should have been bigger")


def test_relabel_by_id(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    rid = log_request(conn, _row())
    relabel_by_id(conn, rid, "local-coder", note=None)
    r = get_by_id(conn, rid)
    assert r is not None and r["relabel_backend"] == "local-coder"


def test_relabel_no_rows_raises(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    with pytest.raises(RelabelError):
        relabel_last(conn, "cloud-large", note=None)
