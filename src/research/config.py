from pathlib import Path
import yaml
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str
    base_url: str
    max_tokens: int


@dataclass
class EmbeddingsConfig:
    model: str
    batch_size: int


@dataclass
class ChunkingConfig:
    size: int
    overlap: int


@dataclass
class RetrievalConfig:
    top_k: int


@dataclass
class StorageConfig:
    db_path: str


@dataclass
class Config:
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    storage: StorageConfig


def load_config(path: str = "configs/config.yaml") -> Config:
    with open(Path(path)) as f:
        raw = yaml.safe_load(f)

    return Config(
        llm=LLMConfig(**raw["llm"]),
        embeddings=EmbeddingsConfig(**raw["embeddings"]),
        chunking=ChunkingConfig(**raw["chunking"]),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        storage=StorageConfig(**raw["storage"]),
    )


config = load_config()
