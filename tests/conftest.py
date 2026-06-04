from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "research.db"


@pytest.fixture
def fake_vectors() -> dict[str, list[float]]:
    return {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0],
        "gamma": [0.0, 0.0, 1.0],
        "mixed": [0.7, 0.7, 0.0],
    }


@pytest.fixture
def fake_embed(fake_vectors: dict[str, list[float]]):
    def _embed(texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                vectors.append(fake_vectors["alpha"])
            elif "beta" in lowered:
                vectors.append(fake_vectors["beta"])
            elif "gamma" in lowered:
                vectors.append(fake_vectors["gamma"])
            else:
                vectors.append(fake_vectors["mixed"])
        return np.asarray(vectors, dtype=np.float32)

    return _embed
