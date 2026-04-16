from __future__ import annotations

import numpy as np
import faiss

from src.config import SETTINGS
from src.ollama_client import OllamaClient
from src.vector_store import load_document_index


def retrieve_top_k_from_doc(
    client: OllamaClient,
    filename: str,
    question: str,
    top_k: int,
    embed_model: str,
) -> list[dict]:
    index, metadata = load_document_index(filename)

    query_vec = client.embed(embed_model, [question])[0]
    query_arr = np.array([query_vec], dtype="float32")
    faiss.normalize_L2(query_arr)

    scores, ids = index.search(query_arr, top_k)
    results: list[dict] = []

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = dict(metadata[idx])
        item["score"] = float(score)
        results.append(item)

    return results


def retrieve_top_k_across_docs(
    client: OllamaClient,
    filenames: list[str],
    question: str,
    top_k: int,
    embed_model: str,
) -> list[dict]:
    all_results: list[dict] = []

    for filename in filenames:
        try:
            doc_results = retrieve_top_k_from_doc(
                client=client,
                filename=filename,
                question=question,
                top_k=top_k,
                embed_model=embed_model,
            )
            all_results.extend(doc_results)
        except FileNotFoundError:
            continue

    all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return all_results[:top_k]