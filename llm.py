"""
llm.py
------
Handles all OpenAI API interactions:
  1. Query rewriting  — makes the user's query retrieval-friendly
  2. RAG synthesis    — grounded answer generation with streaming
  3. Hallucination guard — system prompt enforces context-only answers
"""

import os
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI

from retriever import RetrievedChunk

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_MODEL   = "gpt-3.5-turbo"
MAX_TOKENS   = 1024

RAG_SYSTEM_PROMPT = """You are a knowledgeable and helpful assistant for a hiring agency.
You answer questions strictly based on the provided context excerpts.

Rules:
- If the answer is clearly present in the context, answer concisely and accurately.
- If the context does not contain enough information, say:
  "I don't have enough information in my knowledge base to answer that confidently."
- Never fabricate facts, names, dates, or statistics.
- Always cite the source URL(s) at the end of your answer in the format:
  📎 Source: <url>
- Use bullet points for lists; keep answers readable."""

REWRITE_SYSTEM_PROMPT = """You are a search query optimizer.
Rewrite the user's question into a concise, keyword-rich search query
(1–2 sentences max) optimised for semantic similarity search.
Return ONLY the rewritten query, nothing else."""


# ── Query rewriting ──────────────────────────────────────────────────────────
def rewrite_query(user_query: str, conversation_history: list[dict]) -> str:
    """
    Use the LLM to rephrase the query, incorporating recent conversation context
    so follow-up questions resolve correctly.
    """
    recent_turns = conversation_history[-4:]          # last 2 exchanges
    context_str  = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent_turns
        if m["role"] != "system"
    )
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Conversation so far:\n{context_str}\n\nLatest question: {user_query}"},
    ]
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, max_tokens=128, temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return user_query          # fallback to original


# ── Context builder ──────────────────────────────────────────────────────────
def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {chunk.source})\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


# ── Streaming RAG answer ─────────────────────────────────────────────────────
def stream_rag_answer(
    user_query:           str,
    chunks:               list[RetrievedChunk],
    conversation_history: list[dict],
) -> Generator[str, None, None]:
    """
    Yield answer tokens as a stream.
    conversation_history should be the full list of {role, content} dicts.
    """
    context = build_context(chunks)
    prompt  = (
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n\n"
        f"Answer:"
    )

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        *[m for m in conversation_history if m["role"] != "system"],
        {"role": "user", "content": prompt},
    ]

    try:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:
        yield f"\n\n⚠️ Error: {exc}"


# ── Non-streaming fallback ───────────────────────────────────────────────────
def get_rag_answer(
    user_query:           str,
    chunks:               list[RetrievedChunk],
    conversation_history: list[dict],
) -> str:
    return "".join(stream_rag_answer(user_query, chunks, conversation_history))
