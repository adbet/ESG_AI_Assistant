"""
Albert-native RAG backend.
Uses Albert's /v1/collections, /v1/documents, /v1/search endpoints
so we never call /v1/embeddings directly (which is currently broken).
"""
import requests
import os
from typing import List, Dict
from config import ALBERT_API_KEY, ALBERT_BASE_URL, EMBED_MODEL

BASE = ALBERT_BASE_URL.rstrip("/")
HEADERS = {"Authorization": f"Bearer {ALBERT_API_KEY}"}


def _get(path: str) -> dict:
    r = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, json: dict = None, files=None) -> dict:
    r = requests.post(f"{BASE}{path}", headers=HEADERS, json=json, files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def _delete(path: str):
    r = requests.delete(f"{BASE}{path}", headers=HEADERS, timeout=30)
    r.raise_for_status()


# ── Collections ────────────────────────────────────────────────────────────────

def create_collection(name: str) -> str:
    """Create a collection and return its ID."""
    data = _post("/v1/collections", json={"name": name, "model": EMBED_MODEL})
    return str(data["id"])


def get_or_create_collection(name: str) -> str:
    """Return existing collection ID or create a new one."""
    data = _get("/v1/collections")
    for col in data.get("data", []):
        if col["name"] == name:
            return str(col["id"])
    return create_collection(name)


def delete_collection(collection_id: str):
    _delete(f"/v1/collections/{collection_id}")


def list_collections() -> List[Dict]:
    return _get("/v1/collections").get("data", [])


# ── Documents ──────────────────────────────────────────────────────────────────

def upload_document(collection_id: str, file_bytes: bytes, filename: str) -> str:
    """Upload a PDF to a collection. Albert handles chunking + embedding."""
    files = {"file": (filename, file_bytes, "application/pdf")}
    data = requests.post(
        f"{BASE}/v1/documents",
        headers=HEADERS,
        files=files,
        data={"collection": collection_id},
        timeout=120,
    )
    data.raise_for_status()
    return data.json().get("id", "")


def list_documents(collection_id: str) -> List[Dict]:
    data = _get(f"/v1/documents?collection={collection_id}")
    return data.get("data", [])


# ── Search ─────────────────────────────────────────────────────────────────────

def search(query: str, collection_id: str, k: int = 5) -> List[Dict]:
    """Semantic search over a collection. Returns list of chunk dicts."""
    data = _post("/v1/search", json={
        "collections": [collection_id],
        "prompt": query,
        "k": k,
    })
    results = []
    for hit in data.get("data", []):
        results.append({
            "text": hit.get("content", hit.get("chunk", {}).get("content", "")),
            "source": hit.get("document", {}).get("name", "unknown"),
            "page": hit.get("chunk", {}).get("metadata", {}).get("page", "?"),
            "score": hit.get("score", 0.0),
            "chunk_id": hit.get("chunk", {}).get("id", 0),
        })
    return results
