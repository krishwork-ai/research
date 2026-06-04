from __future__ import annotations

from pathlib import Path

import pytest

from research import ingest, store


def test_detect_source_type_for_pdf_url_and_text(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    assert ingest._detect_source_type(str(pdf)) == "pdf"
    assert ingest._detect_source_type("https://example.test/doc") == "url"
    assert ingest._detect_source_type("plain note") == "text"


def test_chunk_uses_configured_overlap(monkeypatch) -> None:
    monkeypatch.setattr(ingest.config.chunking, "size", 4)
    monkeypatch.setattr(ingest.config.chunking, "overlap", 1)

    assert ingest.chunk("one two three four five six seven") == [
        "one two three four",
        "four five six seven",
    ]


def test_chunk_rejects_overlap_greater_than_size(monkeypatch) -> None:
    monkeypatch.setattr(ingest.config.chunking, "size", 4)
    monkeypatch.setattr(ingest.config.chunking, "overlap", 4)

    with pytest.raises(ValueError, match="must be less"):
        ingest.chunk("one two three four")


def test_add_text_stores_chunks_and_embeddings(monkeypatch, db_path, fake_embed) -> None:
    monkeypatch.setattr(ingest.config.chunking, "size", 3)
    monkeypatch.setattr(ingest.config.chunking, "overlap", 1)
    monkeypatch.setattr(ingest, "embed", fake_embed)
    monkeypatch.setattr(ingest, "init_db", lambda: store.init_db(db_path))

    ingest.add("alpha beta gamma alpha")

    conn = store.init_db(db_path)
    try:
        entries = store.list_entries(conn)
        chunks = store.get_chunks_for_entry(conn, entries[0]["id"])
        embeddings = store.get_all_embeddings(conn)

        assert len(entries) == 1
        assert entries[0]["source_type"] == "text"
        assert [chunk["text"] for chunk in chunks] == [
            "alpha beta gamma",
            "gamma alpha",
        ]
        assert len(embeddings) == 2
    finally:
        conn.close()
