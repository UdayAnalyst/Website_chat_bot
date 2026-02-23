import os
import numpy as np
import faiss
from typing import List
from .config import settings
from .ingest import read_urls, ingest_urls, save_chunks_jsonl

def embed_texts(texts: List[str]) -> np.ndarray:
    provider = settings.EMBEDDINGS_PROVIDER

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing. Create a .env file and set it.")
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        vectors = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = client.embeddings.create(model=settings.OPENAI_EMBED_MODEL, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        return np.array(vectors, dtype=np.float32)

    if provider == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.SBERT_MODEL)
        arr = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        return arr

    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER={provider}")

def main():
    os.makedirs(settings.INDEX_DIR, exist_ok=True)

    urls = read_urls(settings.URLS_PATH)
    chunks = ingest_urls(urls)

    if not chunks:
        raise RuntimeError("No chunks created. Check data/urls.txt and your internet connection.")

    save_chunks_jsonl(chunks, settings.CHUNKS_PATH)

    texts = [c["text"] for c in chunks]
    emb = embed_texts(texts)

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    faiss.write_index(index, settings.FAISS_PATH)

    print(f"✅ Saved FAISS index: {settings.FAISS_PATH}")
    print(f"✅ Saved chunks:     {settings.CHUNKS_PATH}")
    print(f"Chunks indexed: {len(chunks)} | Dim: {emb.shape[1]}")

if __name__ == "__main__":
    main()
