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


def test_clear_requests_empties_and_resets_ids(tmp_path):
    from saint.storage import clear_requests

    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(request_id="a"))
    log_request(conn, _row(request_id="b"))
    assert clear_requests(conn) == 2
    assert conn.execute("SELECT COUNT(*) FROM requests").fetchone() == (0,)
    assert log_request(conn, _row(request_id="c")) == 1  # ids restart


def test_clear_requests_empty_db(tmp_path):
    from saint.storage import clear_requests

    conn = open_db(tmp_path / "log.sqlite")
    assert clear_requests(conn) == 0


def test_schema_v3_fresh_and_upgrade(tmp_path):
    from saint.storage import SCHEMA_VERSION
    assert SCHEMA_VERSION == 3
    conn = open_db(tmp_path / "fresh.sqlite")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    assert {"cache_read_tokens", "cache_write_tokens"} <= cols

    # v2 database with a row upgrades in place, data intact
    import sqlite3
    from importlib import resources
    old = sqlite3.connect(tmp_path / "old.sqlite")
    for m in ("0001_init.sql", "0002_johnny.sql"):
        old.executescript(resources.files("saint.migrations").joinpath(m).read_text())
    old.execute("PRAGMA user_version = 2")
    old.execute(
        "INSERT INTO requests (ts, request_id, model_field, urgency_used, backend_chosen,"
        " success, prompt_storage_mode) VALUES ('t', 'r', 'm', 'normal', 'b', 1, 'full')")
    old.commit()
    old.close()
    conn2 = open_db(tmp_path / "old.sqlite")
    assert schema_version(conn2) == 3
    assert conn2.execute("SELECT COUNT(*) FROM requests").fetchone() == (1,)


def test_log_row_cache_tokens_roundtrip(tmp_path):
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(cache_read_tokens=8712, cache_write_tokens=130))
    r = conn.execute("SELECT cache_read_tokens, cache_write_tokens FROM requests").fetchone()
    assert r == (8712, 130)
    log_request(conn, _row())  # defaults stay NULL
    r2 = conn.execute(
        "SELECT cache_read_tokens, cache_write_tokens FROM requests ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert r2 == (None, None)


def test_fetch_training_rows_excludes_reused_labels(tmp_path):
    from saint.storage import fetch_training_rows
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(request_id="llm", classifier_used="local-chat"))
    log_request(conn, _row(request_id="head", classifier_used="local-embed (embed-head)",
                           prompt_content="head-labeled"))
    log_request(conn, _row(request_id="cache", classifier_used="cache",
                           prompt_content="cache-labeled"))
    log_request(conn, _row(request_id="inh", classifier_used="inherited",
                           prompt_content="inherited-labeled"))
    rows = fetch_training_rows(conn, limit=100)
    prompts = [r[0] for r in rows]
    assert prompts == ["please refactor this"]  # only the LLM-labeled row


def test_classifier_traffic_mix_buckets(tmp_path):
    from saint.storage import classifier_traffic_mix
    conn = open_db(tmp_path / "log.sqlite")
    for used in ("local-embed (embed-head)", "local-embed (embed-head)", "local-chat",
                 "cache", "inherited"):
        log_request(conn, _row(classifier_used=used))
    log_request(conn, _row(classifier_used=None))  # unclassified (pinned) — excluded
    mix = classifier_traffic_mix(conn, limit=100)
    assert mix == {"head": 2, "llm": 1, "reused": 2}


def test_count_training_rows_since(tmp_path):
    from saint.storage import count_training_rows_since
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(classifier_used="local-chat", prompt_content="old prompt"))
    cutoff = "2999-01-01T00:00:00+00:00"
    assert count_training_rows_since(conn, cutoff) == 0
    assert count_training_rows_since(conn, "2000-01-01T00:00:00+00:00") == 1
    # reused/head-labeled rows never count
    log_request(conn, _row(classifier_used="cache", prompt_content="cached prompt"))
    log_request(conn, _row(classifier_used="local-embed (embed-head)", prompt_content="head prompt"))
    assert count_training_rows_since(conn, "2000-01-01T00:00:00+00:00") == 1


def test_traffic_mix_excludes_explain_and_respects_since(tmp_path):
    from saint.storage import classifier_traffic_mix
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(classifier_used="local-chat", model_field="saint-explain"))  # seed/diagnostic
    log_request(conn, _row(classifier_used="local-chat"))
    log_request(conn, _row(classifier_used="local-embed (embed-head)"))
    assert classifier_traffic_mix(conn, 100) == {"head": 1, "llm": 1, "reused": 0}
    # a future `since` excludes everything
    assert classifier_traffic_mix(conn, 100, since="2999-01-01T00:00:00+00:00") == \
        {"head": 0, "llm": 0, "reused": 0}


def test_fetch_training_rows_since_filter(tmp_path):
    from saint.storage import fetch_training_rows
    conn = open_db(tmp_path / "log.sqlite")
    log_request(conn, _row(classifier_used="local-chat"))
    assert len(fetch_training_rows(conn, 10)) == 1
    assert fetch_training_rows(conn, 10, since="2999-01-01T00:00:00+00:00") == []
