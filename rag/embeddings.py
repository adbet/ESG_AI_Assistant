from openai import OpenAI
from typing import List
import numpy as np
import re
import unicodedata
from config import ALBERT_API_KEY, ALBERT_BASE_URL, EMBED_MODEL

_client = OpenAI(api_key=ALBERT_API_KEY, base_url=ALBERT_BASE_URL)
MAX_WORDS = 150
EMBED_DIM = 1024


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    return " ".join(words[:MAX_WORDS])


def get_embeddings(texts: List[str]) -> np.ndarray:
    cleaned = [_clean(t) for t in texts]
    all_embeddings = []
    for text in cleaned:
        if not text:
            all_embeddings.append([0.0] * EMBED_DIM)
            continue
        try:
            response = _client.embeddings.create(model=EMBED_MODEL, input=[text])
            all_embeddings.append(response.data[0].embedding)
        except Exception:
            all_embeddings.append([0.0] * EMBED_DIM)
    return np.array(all_embeddings, dtype=np.float32)


def get_embedding(text: str) -> np.ndarray:
    return get_embeddings([text])[0]