from __future__ import annotations

import streamlit as st

from src.config import SETTINGS, ensure_directories
from src.document_store import delete_uploaded_pdf, save_uploaded_pdf
from src.manifest_store import list_documents, remove_document
from src.ollama_client import OllamaClient
from src.qa_pipeline import (
    answer_question_non_streaming,
    answer_question_streaming,
    index_single_document,
)
from src.ui_helpers import init_session_state
from src.vector_store import delete_document_index, document_index_exists


st.set_page_config(page_title="PDF RAG with Ollama", layout="wide")


@st.cache_resource
def get_ollama_client(base_url: str) -> OllamaClient:
    return OllamaClient(base_url=base_url)


def main() -> None:
    ensure_directories()
    init_session_state()

    st.title("PDF Question Answering with Ollama")
    st.caption("Second-pass version with duplicate detection, streaming, and per-document indexes.")

    with st.sidebar:
        st.header("Settings")

        ollama_url = st.text_input("Ollama URL", value=SETTINGS.ollama_base_url)
        chat_model = st.text_input("Chat model", value=SETTINGS.chat_model)
        embed_model = st.text_input("Embedding model", value=SETTINGS.embed_model)
        keep_alive = st.text_input("Keep model alive", value=SETTINGS.keep_alive)

        st.divider()

        chunk_size = st.slider("Chunk size", 300, 2000, SETTINGS.chunk_size, 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, SETTINGS.chunk_overlap, 10)

        top_k = st.slider("Top-k retrieval", 1, 10, SETTINGS.top_k, 1)
        max_context_chunks = st.slider(
            "Max context chunks sent to model",
            1,
            10,
            SETTINGS.max_context_chunks,
            1,
        )

        streaming_enabled = st.toggle("Stream answer output", value=True)
        st.session_state.streaming_enabled = streaming_enabled

        st.divider()
        st.header("Documents")

        docs = list_documents()
        indexed_names = [d["filename"] for d in docs if d.get("indexed")]

        scope_options = ["All indexed documents"] + indexed_names
        selected_scope = st.selectbox("Query scope", options=scope_options, index=0)
        st.session_state.selected_scope = selected_scope

    client = get_ollama_client(ollama_url)

    ok, msg = client.healthcheck()
    if ok:
        st.success(msg)
    else:
        st.error(msg)
        st.stop()

    st.subheader("Upload PDFs")
    uploads = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploads and st.button("Save and index uploaded PDFs"):
        saved_count = 0
        skipped_count = 0

        for upload in uploads:
            data = upload.read()
            saved, result, path = save_uploaded_pdf(upload.name, data)

            if not saved:
                skipped_count += 1
                st.info(result)
                continue

            file_hash = result
            with st.spinner(f"Indexing {upload.name}..."):
                record = index_single_document(
                    client=client,
                    pdf_path=path,
                    file_hash=file_hash,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embed_model=embed_model,
                )
            saved_count += 1
            st.success(
                f"Indexed {record['filename']} | Pages: {record['pages']} | Chunks: {record['chunks']}"
            )

        if saved_count:
            st.success(f"Saved and indexed {saved_count} new PDF(s).")
        if skipped_count:
            st.warning(f"Skipped {skipped_count} duplicate PDF(s).")

    st.divider()
    st.subheader("Indexed documents")

    docs = list_documents()
    if not docs:
        st.info("No indexed documents yet.")
    else:
        for doc in docs:
            c1, c2, c3, c4 = st.columns([4, 1.5, 1.5, 1.2])
            c1.write(f"**{doc['filename']}**")
            c2.write(f"Pages: {doc.get('pages', 0)}")
            c3.write(f"Chunks: {doc.get('chunks', 0)}")
            if c4.button("Delete", key=f"del-{doc['filename']}"):
                delete_uploaded_pdf(doc["filename"])
                delete_document_index(doc["filename"])
                remove_document(doc["filename"])
                st.warning(f"Deleted {doc['filename']}")
                st.rerun()

    st.divider()
    st.subheader("Ask a question")

    if selected_scope == "All indexed documents":
        selected_files = indexed_names
    else:
        selected_files = [selected_scope]

    question = st.text_area(
        "Question",
        placeholder="Ask a question about the uploaded PDFs...",
        height=120,
    )

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        if not selected_files:
            st.warning("No indexed documents available.")
            st.stop()

        if streaming_enabled:
            with st.spinner("Retrieving and generating answer..."):
                stream, retrieved = answer_question_streaming(
                    client=client,
                    question=question.strip(),
                    selected_files=selected_files,
                    top_k=top_k,
                    max_context_chunks=max_context_chunks,
                    chat_model=chat_model,
                    embed_model=embed_model,
                    keep_alive=keep_alive,
                )

                placeholder = st.empty()
                answer_text = ""
                for token in stream:
                    answer_text += token
                    placeholder.markdown(answer_text)

                st.session_state.last_answer = answer_text
                st.session_state.last_sources = retrieved
        else:
            with st.spinner("Retrieving and generating answer..."):
                answer, retrieved = answer_question_non_streaming(
                    client=client,
                    question=question.strip(),
                    selected_files=selected_files,
                    top_k=top_k,
                    max_context_chunks=max_context_chunks,
                    chat_model=chat_model,
                    embed_model=embed_model,
                    keep_alive=keep_alive,
                )
                st.session_state.last_answer = answer
                st.session_state.last_sources = retrieved

    if st.session_state.last_answer:
        st.subheader("Answer")
        st.write(st.session_state.last_answer)

    if st.session_state.last_sources:
        st.subheader("Retrieved context")
        for i, src in enumerate(st.session_state.last_sources, start=1):
            with st.expander(
                f"{i}. {src.get('source', 'unknown')} | Page {src.get('page', '?')} | Score {src.get('score', 0):.4f}"
            ):
                st.write(src.get("text", ""))


if __name__ == "__main__":
    main()