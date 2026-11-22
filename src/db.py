import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("data/bookscope.db")


@contextmanager
def db():
    """Open a connection, commit if all went well, roll back on error, always close."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row           # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite ignores foreign keys unless asked
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL
                  CHECK (role IN ('author', 'reader', 'publisher', 'admin')),
    verified      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS manuscripts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id   INTEGER NOT NULL REFERENCES users(id),
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS beta_requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    manuscript_id INTEGER NOT NULL REFERENCES manuscripts(id),
    reader_id     INTEGER NOT NULL REFERENCES users(id),
    price         INTEGER NOT NULL CHECK (price > 0),
    status        TEXT NOT NULL DEFAULT 'requested'
                  CHECK (status IN ('requested', 'submitted', 'completed', 'cancelled')),
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (manuscript_id, reader_id)
);

CREATE TABLE IF NOT EXISTS beta_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    beta_request_id INTEGER NOT NULL UNIQUE REFERENCES beta_requests(id),
    plot       INTEGER NOT NULL CHECK (plot       BETWEEN 1 AND 5),
    characters INTEGER NOT NULL CHECK (characters BETWEEN 1 AND 5),
    writing    INTEGER NOT NULL CHECK (writing    BETWEEN 1 AND 5),
    pacing     INTEGER NOT NULL CHECK (pacing     BETWEEN 1 AND 5),
    ending     INTEGER NOT NULL CHECK (ending     BETWEEN 1 AND 5),
    comments     TEXT NOT NULL DEFAULT '',
    submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS escrow_ledger (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    beta_request_id INTEGER NOT NULL UNIQUE REFERENCES beta_requests(id),
    amount          INTEGER NOT NULL CHECK (amount > 0),
    state           TEXT NOT NULL DEFAULT 'held'
                    CHECK (state IN ('held', 'released', 'refunded')),
    updated_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS selections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id    INTEGER NOT NULL UNIQUE REFERENCES users(id),
    publisher_id INTEGER NOT NULL REFERENCES users(id),
    selected_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db():
    """Create tables if they don't exist yet (existing tables and data are kept)."""
    with db() as conn:
        conn.executescript(SCHEMA)