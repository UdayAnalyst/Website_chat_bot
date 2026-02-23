import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from app.config import settings

_sbert_model = None

def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(settings.SBERT_MODEL)
    return _sbert_model

def load_chunks(path: str) -> List[Dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

def embed_query(text: str) -> np.ndarray:
    model = _get_sbert()
    v = model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    return v

def retrieve(index: faiss.Index, chunks: List[Dict], query: str, top_k: int) -> List[Dict]:
    q = embed_query(query)
    scores, ids = index.search(q, top_k)
    results = []
    for i, idx in enumerate(ids[0].tolist()):
        if idx == -1:
            continue
        row = dict(chunks[idx])
        row["score"] = float(scores[0][i])
        results.append(row)
    return results

def build_prompts(user_message: str, sources: List[Dict], user_type: str, intent: str) -> Tuple[str, str]:
    system = (
        "You are a helpful insurance website assistant for a student demo.\n"
        "You MUST answer using ONLY the provided SOURCES.\n"
        "If the answer is not in the sources, say you don't have enough information and provide the closest official page link.\n"
        "Be concise, step-by-step, and include 1-3 citations as plain URLs.\n"
        "Do NOT claim you can access accounts, claim status, or internal systems.\n"
    )

    src_blocks = []
    for s in sources:
        src_blocks.append(
            f"- Title: {s.get('title','')}\n"
            f"  URL: {s.get('source_url')}\n"
            f"  Excerpt: {s.get('text')}\n"
        )

    user = (
        f"User type: {user_type}\n"
        f"Detected intent: {intent}\n"
        f"User question: {user_message}\n\n"
        f"SOURCES:\n" + "\n".join(src_blocks)
    )
    return system, user

def generate_with_groq(system_prompt: str, user_prompt: str) -> str:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing. Set it in .env")

    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)

    resp = client.chat.completions.create(
        model=settings.GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def format_citations(sources: List[Dict], max_cites: int = 3) -> List[Dict]:
    cites = []
    seen = set()
    for s in sources:
        url = s.get("source_url")
        if not url or url in seen:
            continue
        seen.add(url)
        cites.append({"title": s.get("title", "Source"), "url": url})
        if len(cites) >= max_cites:
            break
    return cites
