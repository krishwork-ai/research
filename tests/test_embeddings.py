from __future__ import annotations

import numpy as np

from research import embeddings


def test_similarity_handles_normalised_and_unnormalised_vectors() -> None:
    assert embeddings.similarity(np.array([2.0, 0.0]), np.array([4.0, 0.0])) == 1.0
    assert embeddings.similarity(np.array([0.0, 0.0]), np.array([4.0, 0.0])) == 0.0


def test_vector_search_returns_top_k_ids_in_score_order() -> None:
    chunk_ids = [10, 20, 30]
    chunk_vecs = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )

    assert embeddings.search(np.array([1.0, 0.0]), chunk_ids, chunk_vecs, top_k=2) == [
        10,
        30,
    ]
