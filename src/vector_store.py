from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.config import SETTINGS


def get_doc_dir(filename: str) -> Path:
    stem = Path(filename).stem
    return SETTINGS.index_root_dir / stem


def get_doc_index_file(filename: str) -> Path:
    return get_doc_dir(filename) / "rag.index"


def get_doc_meta_file(filename: str) -> Path:
    return get_doc_dir(filename) / "chunks.json"


def ensure_doc_dir(filename: str) -> Path:
    doc_dir = get_doc_dir(filename)
    doc_dir.mkdir(parents=True, exist_ok=True)
    return doc_dir


def save_document_index(filename: str, embeddings: list[list[float]], metadata: list[dict]) -> None:
    if not embeddings:
        raise ValueError("No embeddings to save.")

    ensure_doc_dir(filename)

    matrix = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(matrix)

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, str(get_doc_index_file(filename)))
    get_doc_meta_file(filename).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_document_index(filename: str) -> tuple[faiss.Index, list[dict]]:
    index_file = get_doc_index_file(filename)
    meta_file = get_doc_meta_file(filename)

    if not index_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Index not found for {filename}")

    index = faiss.read_index(str(index_file))
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    return index, metadata


def document_index_exists(filename: str) -> bool:
    return get_doc_index_file(filename).exists() and get_doc_meta_file(filename).exists()


def delete_document_index(filename: str) -> None:
    doc_dir = get_doc_dir(filename)
    if doc_dir.exists():
        for child in doc_dir.iterdir():
            child.unlink()
        doc_dir.rmdir()