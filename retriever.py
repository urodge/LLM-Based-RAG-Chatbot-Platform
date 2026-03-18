"""
retriever.py
------------
Hybrid retriever: dense FAISS (semantic) + sparse BM25 (keyword).
Results are fused with Reciprocal Rank Fusion (RRF) and optionally
deduplicated before being returned to the LLM.
"""

import os
import pickle
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH  = os.path.join("data", "index.faiss")
CHUNKS_PATH = os.path.join("data", "chunks.pkl")
TOP_K       = 5          # final number of chunks passed to LLM
RRF_K       = 60         # RRF constant (higher → less rank-sensitivity)


@dataclass
class RetrievedChunk:
    text:   str
    source: str
    score:  float


class HybridRetriever:
    """
    Combines:
    - Dense  : FAISS cosine similarity over sentence-transformer embeddings
    - Sparse : BM25 keyword matching (rank-bm25)
    - Fusion : Reciprocal Rank Fusion
    """

    def __init__(
        self,
        index_path:  str = INDEX_PATH,
        chunks_path: str = CHUNKS_PATH,
        top_k:       int = TOP_K,
    ):
        self.top_k = top_k

        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'.\n"
                "Run `python build_index.py` first."
            )
        self.index  = faiss.read_index(index_path)

        # Load raw chunks
        with open(chunks_path, "rb") as f:
            self.chunks: list[dict] = pickle.load(f)

        # Embedding model (same as used during indexing)
        self.model = SentenceTransformer(EMBED_MODEL)

        # Build BM25 corpus
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        print(f"[Retriever] Loaded {len(self.chunks)} chunks | FAISS vectors: {self.index.ntotal}")

    # ── Dense retrieval ───────────────────────────────────────────────────────
    def _dense_search(self, query: str, k: int) -> list[int]:
        vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _, indices = self.index.search(vec, k)
        return indices[0].tolist()

    # ── Sparse retrieval ──────────────────────────────────────────────────────
    def _sparse_search(self, query: str, k: int) -> list[int]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:k]
        return ranked.tolist()

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    @staticmethod
    def _rrf(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, doc_id in enumerate(ranked):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    # ── Public API ────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Return the top-k most relevant chunks for `query` using hybrid search.
        """
        candidate_k = self.top_k * 3          # over-retrieve, then fuse
        dense_ids   = self._dense_search(query, candidate_k)
        sparse_ids  = self._sparse_search(query, candidate_k)

        fused_ids   = self._rrf([dense_ids, sparse_ids])[:self.top_k]

        results = []
        seen_texts = set()
        for idx in fused_ids:
            chunk = self.chunks[idx]
            # Deduplicate near-identical chunks
            key = chunk["text"][:100]
            if key in seen_texts:
                continue
            seen_texts.add(key)
            results.append(
                RetrievedChunk(
                    text=chunk["text"],
                    source=chunk["source"],
                    score=round(1.0 / (fused_ids.index(idx) + 1), 4),
                )
            )
        return results


# ── Singleton loader (cached for Streamlit) ───────────────────────────────────
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
