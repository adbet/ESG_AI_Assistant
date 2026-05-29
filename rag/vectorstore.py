import faiss
import numpy as np
import pickle
import os
from typing import List, Dict
from config import EMBED_DIM


class VectorStore:
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        # Inner-product index — cosine similarity after L2 normalisation
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Dict] = []

    def add(self, chunks: List[Dict], embeddings: np.ndarray):
        """Add document chunks and their embeddings."""
        vecs = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Return the top-k most similar chunks."""
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        k = min(k, len(self.chunks))
        if k == 0:
            return []
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(score)
                results.append(chunk)
        return results

    def save(self, path: str = "vectorstore"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str = "vectorstore"):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

    def __len__(self):
        return len(self.chunks)
