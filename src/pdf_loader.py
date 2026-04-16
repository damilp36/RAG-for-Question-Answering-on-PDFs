from __future__ import annotations

from pathlib import Path
from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            {
                "page": i,
                "text": text.strip(),
            }
        )

    return pages