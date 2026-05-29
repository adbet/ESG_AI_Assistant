from openai import OpenAI
from typing import List, Dict
from config import ALBERT_API_KEY, ALBERT_BASE_URL, CHAT_MODEL

_client = OpenAI(api_key=ALBERT_API_KEY, base_url=ALBERT_BASE_URL)


def _chat(prompt: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    """Generate a cited answer from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are an expert ESG analysis assistant. Answer the question using ONLY the context below.
Cite sources with [Source X] notation after each claim. If the context is insufficient, say so clearly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):"""
    return _chat(prompt, temperature=0.1, max_tokens=1000)


def generate_hypothetical_doc(question: str) -> str:
    """HyDE: generate a hypothetical ESG document passage that would answer the question."""
    prompt = f"""Write a short paragraph (3-5 sentences) from an ESG sustainability report that directly answers this question.
Be specific and factual in style.

Question: {question}

Hypothetical ESG report excerpt:"""
    return _chat(prompt, temperature=0.3, max_tokens=200)


def generate_multiple_queries(question: str) -> List[str]:
    """MQR: generate 3 alternative search queries from one question."""
    prompt = f"""Generate exactly 3 different search queries to find ESG information relevant to this question.
Output one query per line, no numbering, no extra text.

Question: {question}

Queries:"""
    raw = _chat(prompt, temperature=0.5, max_tokens=150)
    queries = [q.strip("•-– ").strip() for q in raw.split("\n") if q.strip()]
    return queries[:3]
