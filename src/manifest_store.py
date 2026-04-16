from __future__ import annotations

import json
from pathlib import Path

from src.config import SETTINGS


def load_manifest() -> dict:
    if not SETTINGS.manifest_file.exists():
        return {"documents": []}
    return json.loads(SETTINGS.manifest_file.read_text(encoding="utf-8"))


def save_manifest(manifest: dict) -> None:
    SETTINGS.manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def get_document_by_hash(file_hash: str) -> dict | None:
    manifest = load_manifest()
    for doc in manifest.get("documents", []):
        if doc.get("sha256") == file_hash:
            return doc
    return None


def get_document_by_name(filename: str) -> dict | None:
    manifest = load_manifest()
    for doc in manifest.get("documents", []):
        if doc.get("filename") == filename:
            return doc
    return None


def upsert_document(record: dict) -> None:
    manifest = load_manifest()
    docs = manifest.get("documents", [])

    replaced = False
    for i, doc in enumerate(docs):
        if doc.get("filename") == record.get("filename"):
            docs[i] = record
            replaced = True
            break

    if not replaced:
        docs.append(record)

    manifest["documents"] = docs
    save_manifest(manifest)


def remove_document(filename: str) -> None:
    manifest = load_manifest()
    manifest["documents"] = [
        d for d in manifest.get("documents", [])
        if d.get("filename") != filename
    ]
    save_manifest(manifest)


def list_documents() -> list[dict]:
    manifest = load_manifest()
    return sorted(manifest.get("documents", []), key=lambda x: x.get("filename", "").lower())