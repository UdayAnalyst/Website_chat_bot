from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # Embeddings provider (use SBERT for Groq setup)
    EMBEDDINGS_PROVIDER: str = os.getenv("EMBEDDINGS_PROVIDER", "sbert").lower()

    # SBERT embeddings (local)
    SBERT_MODEL: str = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")

    # Groq chat generation
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_CHAT_MODEL: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")

    # Chunking
    CHUNK_WORDS: int = int(os.getenv("CHUNK_WORDS", "350"))
    CHUNK_OVERLAP_WORDS: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "80"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # Paths
    URLS_PATH: str = os.getenv("URLS_PATH", "data/urls.txt")
    INDEX_DIR: str = os.getenv("INDEX_DIR", "index")
    FAISS_PATH: str = os.getenv("FAISS_PATH", "index/faiss.index")
    CHUNKS_PATH: str = os.getenv("CHUNKS_PATH", "index/chunks.jsonl")

settings = Settings()
