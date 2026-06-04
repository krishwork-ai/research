from __future__ import annotations

from pathlib import Path

from research.config import load_config


def test_load_config_includes_storage_path(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
llm:
  model: test-chat
  base_url: https://example.test/v1
  max_tokens: 128
embeddings:
  model: test-embed
  batch_size: 4
chunking:
  size: 100
  overlap: 10
retrieval:
  top_k: 3
storage:
  db_path: tmp/test.db
""".strip()
    )

    config = load_config(str(config_file))

    assert config.llm.model == "test-chat"
    assert config.embeddings.batch_size == 4
    assert config.chunking.overlap == 10
    assert config.retrieval.top_k == 3
    assert config.storage.db_path == "tmp/test.db"
