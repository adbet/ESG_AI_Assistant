from typing import List, Dict
import requests
from rag.embeddings import get_embedding
from rag.llm import generate_hypothetical_doc, generate_multiple_queries
from rag.vectorstore import VectorStore
from config import ALBERT_BASE_URL, ALBERT_API_KEY, RERANK_MODEL

_HEADERS = {"Authorization": f"Bearer {ALBERT_API_KEY}"}


def _rerank(query: str, chunks: List[Dict], top_n: int = 5) -> List[Dict]:
    """Rerank chunks using Albert's reranker. Falls back silently."""
    if not chunks:
        return chunks
    try:
        r = requests.post(
            f"{ALBERT_BASE_URL}/rerank",
            headers=_HEADERS,
            json={"model": RERANK_MODEL, "query": query,
                  "documents": [c["text"] for c in chunks], "top_n": top_n},
            timeout=30,
        )
        r.raise_for_status()
        results = sorted(r.json().get("results", []), key=lambda x: x["relevance_score"], reverse=True)
        return [dict(chunks[item["index"]], score=item["relevance_score"]) for item in results[:top_n]]
    except Exception:
        return chunks[:top_n]


def basic_retrieve(query: str, store: VectorStore, k: int = 5) -> List[Dict]:
    candidates = store.search(get_embedding(query), k=k * 2)
    return _rerank(query, candidates, top_n=k)


def hyde_retrieve(query: str, store: VectorStore, k: int = 5) -> List[Dict]:
    hyp_doc = generate_hypothetical_doc(query)
    candidates = store.search(get_embedding(hyp_doc), k=k * 2)
    return _rerank(query, candidates, top_n=k)


def mqr_retrieve(query: str, store: VectorStore, k: int = 5) -> List[Dict]:
    queries = generate_multiple_queries(query) + [query]
    seen: set = set()
    candidates: List[Dict] = []
    for q in queries:
        for hit in store.search(get_embedding(q), k=k):
            key = (hit["source"], hit["page"], hit["chunk_id"])
            if key not in seen:
                seen.add(key)
                candidates.append(hit)
    return _rerank(query, candidates, top_n=k)
