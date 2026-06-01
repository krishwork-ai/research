"""
embeddings.py — vector embeddings via OpenRouter's /v1/embeddings endpoint.

Three public functions:

    embed(texts)              -> np.ndarray  shape (N, D)
    similarity(a, b)          -> float       cosine similarity in [-1, 1]
    search(query, chunk_ids)  -> list[int]   chunk_ids ranked by similarity

The embeddings endpoint is separate from /v1/chat/completions, so this
module makes its own HTTP calls rather than going through llm.complete().

Batching: texts are sent in groups of config.embeddings.batch_size to
avoid hitting request size limits. Batches are sequential (not async) —
for a personal research tool this is fast enough and keeps the code simple.

Vectors are returned as float32 and L2-normalised so that cosine similarity
reduces to a dot product (fast with np.dot / matrix multiply).
"""

from __future__ import annotations

import os
import httpx
import numpy as np

from research.config import config


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class EmbeddingError(RuntimeError):
    """Raised when the OpenRouter embeddings API returns an error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise EmbeddingError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/krishjoshi/research",
        "X-Title": "research",
    }


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row. Zero vectors are left as-is."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a single batch (≤ batch_size texts).
    Returns float32 array of shape (len(texts), embedding_dim).
    """
    payload = {
        "model": config.embeddings.model,
        "input": texts,
    }

    try:
        response = httpx.post(
            f"{config.llm.base_url.rstrip('/')}/embeddings",
            headers=_headers(),
            json=payload,
            timeout=60.0,
        )
    except httpx.RequestError as exc:
        raise EmbeddingError(f"Network error calling embeddings API: {exc}") from exc

    if response.status_code != 200:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise EmbeddingError(
            f"Embeddings API returned {response.status_code}: {detail}",
            status_code=response.status_code,
        )

    try:
        data = response.json()
        # OpenAI-compatible response: {"data": [{"embedding": [...], "index": N}, ...]}
        # Sort by index to guarantee order matches input
        items = sorted(data["data"], key=lambda x: x["index"])
        vectors = [item["embedding"] for item in items]
        return np.array(vectors, dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingError(
            f"Unexpected response shape from embeddings API: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts, batching as needed.

    Parameters
    ----------
    texts:
        List of strings to embed. Empty strings are accepted but will
        produce near-zero vectors (OpenRouter behaviour).

    Returns
    -------
    np.ndarray
        Float32 array of shape (len(texts), embedding_dim), L2-normalised.
        Rows are in the same order as `texts`.

    Raises
    ------
    EmbeddingError
        On API errors or unexpected response shape.
    ValueError
        If texts is empty.
    """
    if not texts:
        raise ValueError("embed() requires at least one text")

    batch_size = config.embeddings.batch_size
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    parts: list[np.ndarray] = []
    for batch in batches:
        parts.append(_embed_batch(batch))

    matrix = np.vstack(parts) if len(parts) > 1 else parts[0]
    return _normalise(matrix)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.

    Both vectors should already be L2-normalised (as returned by embed()),
    in which case this is just a dot product. If unnormalised vectors are
    passed the function normalises them first so the result is always valid.

    Returns a float in [-1.0, 1.0]. Identical vectors → 1.0.
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a / norm_a, b / norm_b))


def search(
    query_vec: np.ndarray,
    chunk_ids: list[int],
    chunk_vecs: np.ndarray,
    top_k: int | None = None,
) -> list[int]:
    """
    Rank chunk_ids by cosine similarity to query_vec.

    Parameters
    ----------
    query_vec:
        1-D float array, the embedded query (as returned by embed()).
    chunk_ids:
        List of chunk ids (integers) — must be the same length as chunk_vecs.
    chunk_vecs:
        2-D float array of shape (N, D) — one row per chunk, in the same
        order as chunk_ids.
    top_k:
        How many results to return. Defaults to config.retrieval.top_k.

    Returns
    -------
    list[int]
        chunk_ids sorted by descending similarity, up to top_k entries.

    Raises
    ------
    ValueError
        If chunk_ids and chunk_vecs have different lengths, or are empty.
    """
    if len(chunk_ids) != len(chunk_vecs):
        raise ValueError(
            f"chunk_ids ({len(chunk_ids)}) and chunk_vecs ({len(chunk_vecs)}) "
            "must be the same length"
        )
    if len(chunk_ids) == 0:
        return []

    k = top_k if top_k is not None else config.retrieval.top_k

    q = np.asarray(query_vec, dtype=np.float32).ravel()
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    # Matrix multiply: scores[i] = dot(q, chunk_vecs[i])
    # chunk_vecs rows are already normalised (from embed()), so this is cosine sim.
    vecs = np.asarray(chunk_vecs, dtype=np.float32)
    scores = vecs @ q  # shape (N,)

    # argsort descending, take top_k
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunk_ids[i] for i in top_indices]
