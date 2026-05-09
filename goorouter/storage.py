from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path


SCHEMA_VERSION = 1


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
