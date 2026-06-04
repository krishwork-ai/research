"""
retrieve.py — query retrieval layer.

Single public function:

    search(query, top_k=None) -> list[dict]

Takes a natural-language query string, embeds it, scores it against
every stored chunk vector, and returns the top-k chunks with full
source provenance — ready to hand straight to the synthesis step.

Each returned dict has keys (from store.get_chunks_with_sources):
    chunk_id    int
    text        str   — the raw chunk text
    chunk_index int   — position within its source document
    source      str   — file path, URL, or raw text note
    source_type str   — 'pdf' | 'url' | 'text'
    added_at    str   — ISO timestamp

Returns [] if the store is empty (no embeddings yet).

Raises:
    ValueError      — if query is empty or whitespace-only
    EmbeddingError  — if the OpenRouter embeddings call fails
"""

from __future__ import annotations

import numpy as np

from research.config import config
from research.embeddings import embed, search as vector_search
from research.store import get_all_embeddings, get_chunks_with_sources, init_db


def search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Retrieve the top-k chunks most relevant to a natural-language query.

    Parameters
    ----------
    query:
        The question or search string to look up.
    top_k:
        Number of results to return. Defaults to config.retrieval.top_k.

    Returns
    -------
    list[dict]
        Chunks ranked by descending cosine similarity, each with keys:
        chunk_id, text, chunk_index, source, source_type, added_at.
        Empty list if the store contains no embeddings.

    Raises
    ------
    ValueError
        If query is empty or whitespace-only.
    EmbeddingError
        If the embedding API call fails.
    """
    query = query.strip()
    if not query:
        raise ValueError("search query must not be empty")

    k = top_k if top_k is not None else config.retrieval.top_k

    conn = init_db()
    try:
        all_embeddings = get_all_embeddings(conn)

        if not all_embeddings:
            return []

        # Unzip into parallel lists for vector_search
        chunk_ids, chunk_vecs = zip(*all_embeddings)
        chunk_ids = list(chunk_ids)
        chunk_matrix = np.stack(chunk_vecs)  # shape (N, D)

        # Embed the query — embed() returns shape (1, D)
        query_vec = embed([query])[0]

        # Rank by similarity
        ranked_ids = vector_search(query_vec, chunk_ids, chunk_matrix, top_k=k)

        # Fetch text + provenance in ranked order
        return get_chunks_with_sources(conn, ranked_ids)

    finally:
        conn.close()
