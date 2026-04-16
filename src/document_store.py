from __future__ import annotations

import hashlib
from pathlib import Path

from src.config import SETTINGS
from src.manifest_store import get_document_by_hash


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_uploaded_pdf(filename: str, data: bytes) -> tuple[bool, str, Path | None]:
    """
    Returns:
        (saved, message, path)
    """
    file_hash = sha256_bytes(data)
    existing = get_document_by_hash(file_hash)
    if existing:
        return False, f"Duplicate file detected. Already stored as {existing['filename']}.", None

    safe_name = Path(filename).name
    output_path = SETTINGS.uploads_dir / safe_name
    output_path.write_bytes(data)
    return True, file_hash, output_path


def delete_uploaded_pdf(filename: str) -> bool:
    target = SETTINGS.uploads_dir / Path(filename).name
    if target.exists():
        target.unlink()
        return True
    return False