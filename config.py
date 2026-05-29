import os
from dotenv import load_dotenv

load_dotenv()

ALBERT_API_KEY = os.getenv("ALBERT_API_KEY")
ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"

# Models
EMBED_MODEL   = "BAAI/bge-m3"
CHAT_MODEL    = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"

# Chunking
CHUNK_SIZE = 300       # words per chunk (reduced to avoid InternalServerError)
CHUNK_OVERLAP = 30     # word overlap between chunks

# Retrieval
TOP_K = 5
EMBED_DIM = 1024       # BGE-M3 output dimension
EMBED_BATCH_SIZE = 8   # small batches to avoid server errors
