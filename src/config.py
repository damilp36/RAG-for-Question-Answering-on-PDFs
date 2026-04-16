from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    uploads_dir: Path = data_dir / "uploads"
    index_root_dir: Path = data_dir / "index" / "docs"
    metadata_dir: Path = data_dir / "metadata"

    manifest_file: Path = metadata_dir / "document_manifest.json"

    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "4"))


SETTINGS = Settings()


def ensure_directories() -> None:
    SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.uploads_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.index_root_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.metadata_dir.mkdir(parents=True, exist_ok=True)