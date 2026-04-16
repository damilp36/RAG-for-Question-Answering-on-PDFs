from __future__ import annotations

from pathlib import Path

from src.embeddings import embed_chunks
from src.manifest_store import upsert_document
from src.ollama_client import OllamaClient
from src.pdf_loader import extract_pdf_text
from src.prompt_builder import build_qa_messages
from src.retriever import retrieve_top_k_across_docs, retrieve_top_k_from_doc
from src.text_splitter import pages_to_chunks
from src.vector_store import save_document_index


def index_single_document(
    client: OllamaClient,
    pdf_path: Path,
    file_hash: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
) -> dict:
    pages = extract_pdf_text(pdf_path)
    chunks = pages_to_chunks(
        pages=pages,
        source_name=pdf_path.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(f"No text chunks created for {pdf_path.name}")

    texts = [c["text"] for c in chunks]
    embeddings = client.embed(embed_model, texts)

    save_document_index(pdf_path.name, embeddings, chunks)

    record = {
        "filename": pdf_path.name,
        "sha256": file_hash,
        "pages": len(pages),
        "chunks": len(chunks),
        "indexed": True,
    }
    upsert_document(record)
    return record


def answer_question_non_streaming(
    client: OllamaClient,
    question: str,
    selected_files: list[str],
    top_k: int,
    max_context_chunks: int,
    chat_model: str,
    embed_model: str,
    keep_alive: str,
) -> tuple[str, list[dict]]:
    if len(selected_files) == 1:
        retrieved = retrieve_top_k_from_doc(
            client=client,
            filename=selected_files[0],
            question=question,
            top_k=top_k,
            embed_model=embed_model,
        )
    else:
        retrieved = retrieve_top_k_across_docs(
            client=client,
            filenames=selected_files,
            question=question,
            top_k=top_k,
            embed_model=embed_model,
        )

    messages = build_qa_messages(question, retrieved[:max_context_chunks])
    answer = client.chat(chat_model, messages, keep_alive=keep_alive)
    return answer, retrieved


def answer_question_streaming(
    client: OllamaClient,
    question: str,
    selected_files: list[str],
    top_k: int,
    max_context_chunks: int,
    chat_model: str,
    embed_model: str,
    keep_alive: str,
):
    if len(selected_files) == 1:
        retrieved = retrieve_top_k_from_doc(
            client=client,
            filename=selected_files[0],
            question=question,
            top_k=top_k,
            embed_model=embed_model,
        )
    else:
        retrieved = retrieve_top_k_across_docs(
            client=client,
            filenames=selected_files,
            question=question,
            top_k=top_k,
            embed_model=embed_model,
        )

    messages = build_qa_messages(question, retrieved[:max_context_chunks])
    stream = client.stream_chat(chat_model, messages, keep_alive=keep_alive)
    return stream, retrieved