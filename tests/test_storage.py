import sqlite3
from pathlib import Path

import pytest

from goorouter.storage import open_db, schema_version


def test_open_db_creates_schema(tmp_path: Path):
    db = tmp_path / "log.sqlite"
    conn = open_db(db)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'")
    assert cursor.fetchone() is not None
    assert schema_version(conn) == 1


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
    assert schema_version(conn) == 1
