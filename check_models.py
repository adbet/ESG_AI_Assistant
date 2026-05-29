"""Run this to find the correct Albert API model names."""
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ALBERT_API_KEY"),
    base_url="https://albert.api.etalab.gouv.fr/v1"
)

print("=== Available Models ===")
for m in client.models.list():
    print(f"  {m.id}  |  type: {getattr(m, 'type', 'unknown')}")

print("\n=== Embedding model candidates ===")
models = list(client.models.list())
embed_candidates = [m for m in models if any(
    kw in m.id.lower() for kw in ["bge", "embed", "e5", "minilm", "multilingual"]
)]
print([m.id for m in embed_candidates])
