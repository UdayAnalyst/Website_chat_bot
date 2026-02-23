import re
import json
import time
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from .config import settings

USER_AGENT = "Mozilla/5.0 (compatible; UConn-RAG-Demo/1.0)"

def read_urls(path: str) -> List[str]:
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)
    return urls

def fetch_html(url: str, timeout: int = 20) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.text

def html_to_text(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return {"title": title, "text": text}

def chunk_words(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks

def guess_section(url: str) -> str:
    u = url.lower()
    if "claim" in u: return "claims"
    if "pay" in u or "bill" in u: return "billing"
    if "agent" in u: return "agent"
    if "contact" in u: return "contact"
    return "general"

def ingest_urls(urls: List[str], sleep_s: float = 0.5) -> List[Dict]:
    all_chunks = []
    chunk_id = 0

    for url in tqdm(urls, desc="Fetching pages"):
        try:
            html = fetch_html(url)
            doc = html_to_text(html)
            section = guess_section(url)

            chunks = chunk_words(doc["text"], settings.CHUNK_WORDS, settings.CHUNK_OVERLAP_WORDS)
            for c in chunks:
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "source_url": url,
                    "title": doc["title"],
                    "section": section,
                    "text": c
                })
                chunk_id += 1
            time.sleep(sleep_s)
        except Exception as e:
            print(f"[WARN] Failed {url}: {e}")
    return all_chunks

def save_chunks_jsonl(chunks: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
