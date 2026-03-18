"""
app.py
------
Streamlit conversational UI for the LLM-RAG Hiring Assistant.

Run:
    streamlit run app.py
"""

import streamlit as st

from llm import get_rag_answer, rewrite_query, stream_rag_answer
from retriever import get_retriever

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hiring Assistant · RAG Chatbot",
    page_icon="🤝",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main { background: #0f1117; }
    
    /* Chat message bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 4px 0;
    }

    /* Source expander */
    .streamlit-expanderHeader {
        font-size: 0.78rem;
        color: #9ca3af;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Title */
    h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    show_sources = st.toggle("Show retrieved sources", value=True)
    show_rewrite = st.toggle("Show rewritten query",   value=False)
    top_k        = st.slider("Chunks retrieved (top-k)", min_value=2, max_value=10, value=5)

    st.divider()
    st.markdown("### About")
    st.markdown(
        "A **RAG chatbot** for hiring agency Q&A.  \n"
        "Answers are grounded in your indexed knowledge base — no hallucinations."
    )
    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.rewrite_log   = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
SYSTEM_MSG = {
    "role":    "system",
    "content": "You are a helpful assistant for a hiring agency.",
}

if "messages"    not in st.session_state:
    st.session_state.messages    = []
if "rewrite_log" not in st.session_state:
    st.session_state.rewrite_log = []   # parallel list to messages (user turns only)

# ── Load retriever (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base …")
def load_retriever():
    return get_retriever()

try:
    retriever = load_retriever()
    index_ready = True
except FileNotFoundError as exc:
    st.error(str(exc))
    index_ready = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤝 Hiring Assistant")
st.caption("Ask anything about our hiring process, roles, candidates, or policies.")
st.divider()

# ── Render existing chat ──────────────────────────────────────────────────────
rewrite_idx = 0
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show retrieved sources below assistant messages (stored in metadata)
        if msg["role"] == "assistant" and show_sources:
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📎 {len(sources)} source chunk(s) used"):
                    for src in sources:
                        st.markdown(f"**Source:** `{src['source']}`")
                        st.markdown(f"> {src['text'][:300]}…")
                        st.divider()

        # Show rewritten query below user messages
        if msg["role"] == "user" and show_rewrite and rewrite_idx < len(st.session_state.rewrite_log):
            rw = st.session_state.rewrite_log[rewrite_idx]
            if rw != msg["content"]:
                st.caption(f"🔍 Rewritten query: *{rw}*")
            rewrite_idx += 1

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about hiring, roles, or candidates …", disabled=not index_ready):

    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Rewrite query for better retrieval
    full_history = [SYSTEM_MSG] + st.session_state.messages
    rewritten    = rewrite_query(prompt, full_history)
    st.session_state.rewrite_log.append(rewritten)

    if show_rewrite and rewritten != prompt:
        with st.chat_message("user"):           # append under the same turn visually
            st.caption(f"🔍 Rewritten query: *{rewritten}*")

    # 3. Retrieve
    with st.spinner("Searching knowledge base …"):
        chunks = retriever.retrieve(rewritten)
        chunks = chunks[:top_k]                 # respect sidebar setting

    # 4. Stream answer
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for token in stream_rag_answer(prompt, chunks, full_history):
            full_response += token
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        # Show sources inline
        if show_sources and chunks:
            with st.expander(f"📎 {len(chunks)} source chunk(s) used"):
                for chunk in chunks:
                    st.markdown(f"**Source:** `{chunk.source}`")
                    st.markdown(f"> {chunk.text[:300]}…")
                    st.divider()

    # 5. Persist assistant message with source metadata
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "sources": [{"text": c.text, "source": c.source} for c in chunks],
    })
