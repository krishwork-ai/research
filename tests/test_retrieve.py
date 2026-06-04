from __future__ import annotations

import numpy as np
import pytest

from research import retrieve, store


def test_search_retrieves_ranked_chunks(monkeypatch, db_path) -> None:
    conn = store.init_db(db_path)
    try:
        entry_id = store.insert_entry(conn, "notes", "text")
        chunk_ids = store.insert_chunks(conn, entry_id, ["alpha topic", "beta topic"])
        store.insert_embeddings(
            conn,
            chunk_ids,
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            model="test-embed",
        )
    finally:
        conn.close()

    monkeypatch.setattr(retrieve, "init_db", lambda: store.init_db(db_path))
    monkeypatch.setattr(
        retrieve,
        "embed",
        lambda texts: np.asarray([[0.9, 0.1]], dtype=np.float32),
    )

    results = retrieve.search("alpha", top_k=2)

    assert [result["text"] for result in results] == ["alpha topic", "beta topic"]


def test_search_returns_empty_when_store_has_no_embeddings(monkeypatch, db_path) -> None:
    store.init_db(db_path).close()
    monkeypatch.setattr(retrieve, "init_db", lambda: store.init_db(db_path))

    assert retrieve.search("alpha") == []


def test_search_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        retrieve.search("   ")
