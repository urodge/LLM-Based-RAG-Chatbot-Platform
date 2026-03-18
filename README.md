# LLM-Based RAG Chatbot Platform

A full-stack **Retrieval-Augmented Generation (RAG)** system with hybrid semantic + keyword search, GPT-3.5 inference, query rewriting, streaming responses, and a Streamlit conversational UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green)
![FAISS](https://img.shields.io/badge/FAISS-vector--search-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

---

## Architecture

```
User Query
    │
    ▼
┌────────────────────────┐
│  Query Rewriter         │  ← GPT-3.5 rewrites query for better retrieval
└────────┬───────────────┘
         │
    ┌────┴──────────────────────────────────┐
    │              Hybrid Retrieval          │
    │  ┌──────────────┐  ┌───────────────┐  │
    │  │ Dense (FAISS) │  │ Sparse (BM25) │  │
    │  │ cosine sim    │  │ keyword match │  │
    │  └──────┬───────┘  └──────┬────────┘  │
    │         └────── RRF ──────┘           │
    └────────────────┬──────────────────────┘
                     │  Top-k chunks + sources
                     ▼
         ┌───────────────────────┐
         │   GPT-3.5 Turbo        │  ← Grounded answer synthesis (streaming)
         │   RAG system prompt    │     with hallucination guard
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Streamlit UI         │  ← Chat history, source viewer, settings
         └───────────────────────┘
```

**Offline Indexing Pipeline:**
```
URLs (urls.txt)
    → BeautifulSoup scraping
    → Text cleaning + overlapping chunking (512 tok, 128 stride)
    → Sentence Transformer embeddings (all-MiniLM-L6-v2)
    → FAISS IndexFlatIP (cosine similarity)
    → BM25 index built in-memory at runtime
```

---

## Tech Stack

| Component        | Tool                                      |
|-----------------|-------------------------------------------|
| Embedding model  | `sentence-transformers/all-MiniLM-L6-v2`  |
| Dense search     | FAISS `IndexFlatIP` (cosine similarity)   |
| Sparse search    | BM25 via `rank-bm25`                      |
| Fusion           | Reciprocal Rank Fusion (RRF)              |
| LLM              | OpenAI GPT-3.5-Turbo (streaming)          |
| Web scraping     | BeautifulSoup4 + requests                 |
| UI               | Streamlit                                 |
| Language         | Python 3.10+                              |

---

## Project Structure

```
├── app.py              # Streamlit UI — chat interface, source viewer, settings
├── build_index.py      # Offline pipeline: scrape → chunk → embed → FAISS index
├── retriever.py        # Hybrid BM25 + FAISS retrieval with RRF fusion
├── llm.py              # Query rewriting, RAG prompt, streaming GPT-3.5 call
├── urls.txt            # One URL per line — pages to index
├── data/
│   ├── index.faiss     # FAISS vector index (generated)
│   └── chunks.pkl      # Raw text chunks + source metadata (generated)
├── requirements.txt
├── .env.example        # Copy to .env and add your OpenAI key
└── README.md
```

---

## Setup

```bash
git clone https://github.com/urodge/LLM-RAG-Chatbot
cd LLM-RAG-Chatbot
pip install -r requirements.txt

# 1. Set your OpenAI key
cp .env.example .env
# Edit .env → OPENAI_API_KEY=sk-...

# 2. Add URLs to index
nano urls.txt   # one URL per line

# 3. Build the index (run once, or after adding new URLs)
python build_index.py

# 4. Launch the app
streamlit run app.py
```

---

## Key Design Decisions

**Hybrid Search (BM25 + FAISS)**  
Pure semantic search misses exact-match queries (names, job IDs, codes). BM25 catches these; FAISS handles conceptual similarity. RRF fusion combines both ranked lists without needing to tune score scales.

**Query Rewriting**  
Follow-up questions like "What about salary?" are ambiguous without conversation history. The rewriter injects context, turning them into self-contained retrieval queries.

**Hallucination Guard**  
The RAG system prompt explicitly instructs the model to admit uncertainty when context is insufficient, preventing confident wrong answers.

**Streaming Responses**  
GPT-3.5 answers are streamed token-by-token for a responsive UX — no waiting for the full response to generate.

**Source Citations**  
Every answer links back to the source URL(s) it was grounded in, making the system auditable.

---

## License

MIT
