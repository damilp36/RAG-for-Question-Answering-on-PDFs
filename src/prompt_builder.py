from __future__ import annotations


def build_qa_messages(question: str, retrieved_chunks: list[dict]) -> list[dict[str, str]]:
    context_parts = []

    for chunk in retrieved_chunks:
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        context_parts.append(f"[Source: {source}, Page: {page}]\n{text}")

    context_block = "\n\n".join(context_parts)

    system_message = (
        "You answer questions strictly from the provided PDF context. "
        "If the answer is not present in the context, say so clearly. "
        "Always cite source filename and page number."
    )

    user_message = f"""Context:
{context_block}

Question:
{question}
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]