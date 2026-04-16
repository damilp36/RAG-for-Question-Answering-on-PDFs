from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(end - chunk_overlap, 0)

    return chunks


def pages_to_chunks(
    pages: list[dict],
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    records: list[dict] = []

    for page_obj in pages:
        page_num = page_obj["page"]
        page_text = page_obj["text"]

        for idx, chunk in enumerate(
            chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            records.append(
                {
                    "source": source_name,
                    "page": page_num,
                    "chunk_id": f"{source_name}-p{page_num}-c{idx}",
                    "text": chunk,
                }
            )

    return records