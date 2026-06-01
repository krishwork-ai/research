"""
store.py — SQLite persistence for the research system.

Three tables:
  entries    — one row per source (pdf path, url, or raw text)
  chunks     — one row per text chunk, FK → entries
  embeddings — one row per chunk,  FK → chunks, vector stored as blob

Design notes:
  - embeddings live in a separate table so the model can be swapped
    without touching chunk text (just drop + refill embeddings).
  - vectors are stored as raw float32 blobs; numpy is used for
    serialisation/deserialisation — no extension required.
  - all public functions receive/return plain Python types so callers
    never touch sqlite3 directly.
"""

from __future__ import annotations

import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from research.config import config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_PATH = (
    Path(config.storage.db_path)
    if hasattr(config, "storage")
    else Path("data/research.db")
)


def _vec_to_blob(vec: list[float] | np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Return a connection with WAL mode and foreign-key enforcement on."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,          -- file path, url, or first 120 chars of raw text
    source_type TEXT    NOT NULL,          -- 'pdf' | 'url' | 'text'
    added_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,          -- 0-based position within the source
    text        TEXT    NOT NULL,
    added_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id    INTEGER NOT NULL UNIQUE REFERENCES chunks(id) ON DELETE CASCADE,
    model       TEXT    NOT NULL,          -- embedding model name, e.g. BAAI/bge-small-en-v1.5
    vector      BLOB    NOT NULL,          -- float32 bytes
    added_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_entry    ON chunks(entry_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(chunk_id);
"""


def init_db(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = get_connection(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def insert_entry(
    conn: sqlite3.Connection,
    source: str,
    source_type: str,
) -> int:
    """Insert a source record. Returns the new entry id."""
    cur = conn.execute(
        "INSERT INTO entries (source, source_type, added_at) VALUES (?, ?, ?)",
        (source, source_type, _now()),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_entry(conn: sqlite3.Connection, entry_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()


def list_entries(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM entries ORDER BY added_at DESC").fetchall()


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


def insert_chunks(
    conn: sqlite3.Connection,
    entry_id: int,
    texts: list[str],
) -> list[int]:
    """
    Bulk-insert chunks for an entry.
    Returns list of new chunk ids in order.
    """
    now = _now()
    rows = [(entry_id, idx, text, now) for idx, text in enumerate(texts)]
    conn.executemany(
        "INSERT INTO chunks (entry_id, chunk_index, text, added_at) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    # Fetch back the ids we just inserted (lowest rowid for this entry_id)
    rows_back = conn.execute(
        "SELECT id FROM chunks WHERE entry_id = ? ORDER BY chunk_index",
        (entry_id,),
    ).fetchall()
    return [r["id"] for r in rows_back]


def get_chunk(conn: sqlite3.Connection, chunk_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()


def get_chunks_for_entry(conn: sqlite3.Connection, entry_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE entry_id = ? ORDER BY chunk_index",
        (entry_id,),
    ).fetchall()


def iter_all_chunks(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    """Yield every chunk row — used during search to pair with embeddings."""
    yield from conn.execute("SELECT * FROM chunks ORDER BY id")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def insert_embeddings(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
    vectors: list[list[float]] | np.ndarray,
    model: str,
) -> None:
    """
    Bulk-insert embedding rows.
    chunk_ids and vectors must be the same length and in the same order.
    Uses INSERT OR REPLACE so re-embedding after a model swap is safe.
    """
    now = _now()
    rows = [
        (chunk_id, model, _vec_to_blob(vec), now)
        for chunk_id, vec in zip(chunk_ids, vectors)
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO embeddings (chunk_id, model, vector, added_at)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def get_embedding(conn: sqlite3.Connection, chunk_id: int) -> np.ndarray | None:
    """Return the vector for a chunk, or None if it hasn't been embedded."""
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return _blob_to_vec(row["vector"]) if row else None


def get_all_embeddings(
    conn: sqlite3.Connection,
) -> list[tuple[int, np.ndarray]]:
    """
    Return [(chunk_id, vector), ...] for every stored embedding.
    Used by the retrieval layer to build the search matrix.
    """
    rows = conn.execute(
        "SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id"
    ).fetchall()
    return [(r["chunk_id"], _blob_to_vec(r["vector"])) for r in rows]


# ---------------------------------------------------------------------------
# Convenience: full retrieval for search results
# ---------------------------------------------------------------------------


def get_chunks_with_sources(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
) -> list[dict]:
    """
    Given a list of chunk ids (e.g. the top-k from a similarity search),
    return a list of dicts with chunk text + source provenance.

    Returned keys: chunk_id, text, chunk_index, source, source_type, added_at
    """
    if not chunk_ids:
        return []

    placeholders = ", ".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT
            c.id          AS chunk_id,
            c.text        AS text,
            c.chunk_index AS chunk_index,
            e.source      AS source,
            e.source_type AS source_type,
            c.added_at    AS added_at
        FROM chunks c
        JOIN entries e ON e.id = c.entry_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()

    # Return in the same order the caller provided (similarity-ranked)
    by_id = {r["chunk_id"]: dict(r) for r in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


# ---------------------------------------------------------------------------
# Teardown (tests / dev)
# ---------------------------------------------------------------------------


def drop_all(conn: sqlite3.Connection) -> None:
    """Wipe all application tables. Useful for test teardown."""
    conn.executescript("""
        DROP TABLE IF EXISTS embeddings;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS entries;
    """)
    conn.commit()
