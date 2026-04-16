from __future__ import annotations

from src.config import SETTINGS
from src.ollama_client import OllamaClient


def embed_chunks(client: OllamaClient, chunks: list[dict]) -> list[list[float]]:
    texts = [c["text"] for c in chunks]
    return client.embed(SETTINGS.embed_model, texts)