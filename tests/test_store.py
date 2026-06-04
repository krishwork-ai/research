from __future__ import annotations

import numpy as np

from research import store


def test_store_round_trip_entry_chunks_embeddings(db_path) -> None:
    conn = store.init_db(db_path)
    try:
        entry_id = store.insert_entry(conn, "alpha note", "text")
        chunk_ids = store.insert_chunks(conn, entry_id, ["alpha one", "alpha two"])
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        store.insert_embeddings(conn, chunk_ids, vectors, model="test-embed")

        assert store.get_entry(conn, entry_id)["source"] == "alpha note"
        assert [row["text"] for row in store.get_chunks_for_entry(conn, entry_id)] == [
            "alpha one",
            "alpha two",
        ]
        assert np.allclose(store.get_embedding(conn, chunk_ids[0]), vectors[0])
        assert [chunk_id for chunk_id, _ in store.get_all_embeddings(conn)] == chunk_ids

        rows = store.get_chunks_with_sources(conn, [chunk_ids[1], chunk_ids[0]])
        assert [row["chunk_id"] for row in rows] == [chunk_ids[1], chunk_ids[0]]
        assert rows[0]["source"] == "alpha note"
    finally:
        conn.close()


def test_drop_all_removes_application_tables(db_path) -> None:
    conn = store.init_db(db_path)
    try:
        store.insert_entry(conn, "alpha note", "text")
        store.drop_all(conn)

        table_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"entries", "chunks", "embeddings"}.isdisjoint(table_names)
    finally:
        conn.close()
