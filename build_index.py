"""
build_index.py
--------------
Offline pipeline: scrape → clean → chunk → embed → FAISS index.

Run once (or whenever your knowledge base changes):
    python build_index.py --urls urls.txt --output data/
"""

import argparse
import json
import os
import pickle
import re
import time

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 512          # tokens (approx chars / 4)
CHUNK_OVERLAP = 128          # stride overlap
OUTPUT_DIR    = "data"
INDEX_PATH    = os.path.join(OUTPUT_DIR, "index.faiss")
CHUNKS_PATH   = os.path.join(OUTPUT_DIR, "chunks.pkl")   # raw text chunks + metadata
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Scraping ──────────────────────────────────────────────────────────────
def scrape_url(url: str, delay: float = 1.0) -> str:
    """Fetch and extract clean text from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (RAG-Chatbot-Indexer/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excess newlines
        text = re.sub(r"[ \t]{2,}", " ", text)    # collapse excess spaces
        time.sleep(delay)
        return text.strip()
    except Exception as exc:
        print(f"  [WARN] Could not scrape {url}: {exc}")
        return ""


def load_urls(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# ── 2. Chunking ──────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping windows (by character approximation of tokens).
    Each chunk is a dict with 'text' and 'source' metadata.
    """
    # Approx: 1 token ≈ 4 chars
    char_size    = chunk_size * 4
    char_overlap = overlap * 4

    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + char_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:                       # skip tiny fragments
            chunks.append({"text": chunk, "source": source})
        start += char_size - char_overlap

    return chunks


# ── 3. Embedding + FAISS ─────────────────────────────────────────────────────
def build_faiss_index(chunks: list[dict], model: SentenceTransformer) -> faiss.IndexFlatIP:
    """Embed all chunks and build a normalised inner-product FAISS index (= cosine sim)."""
    print(f"\nEmbedding {len(chunks)} chunks …")
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)               # cosine similarity via L2-normalised vectors
    index.add(embeddings)
    return index


# ── 4. Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build RAG FAISS index from URLs or local text files.")
    parser.add_argument("--urls",   default="urls.txt",  help="File with one URL per line")
    parser.add_argument("--output", default=OUTPUT_DIR,  help="Output directory for index + chunks")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    index_path  = os.path.join(args.output, "index.faiss")
    chunks_path = os.path.join(args.output, "chunks.pkl")

    # -- Scrape
    if not os.path.exists(args.urls):
        print(f"[ERROR] URL file not found: {args.urls}")
        print("Create a file 'urls.txt' with one URL per line and re-run.")
        return

    urls = load_urls(args.urls)
    print(f"Scraping {len(urls)} URL(s) …")
    all_chunks: list[dict] = []
    for url in urls:
        print(f"  → {url}")
        raw_text = scrape_url(url)
        chunks   = chunk_text(raw_text, source=url)
        print(f"     {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[ERROR] No text scraped. Check your URLs.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")

    # -- Embed + index
    model = SentenceTransformer(EMBED_MODEL)
    index = build_faiss_index(all_chunks, model)

    # -- Persist
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n✅  Index saved  → {index_path}")
    print(f"✅  Chunks saved → {chunks_path}")
    print(f"    Total vectors: {index.ntotal}")


if __name__ == "__main__":
    main()
