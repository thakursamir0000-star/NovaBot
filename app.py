import html
import logging
import os
import tempfile
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

import llm
from ingest import chunk_with_overlap, document_id, extract_book_title, extract_text_by_page, safe_overlap
from retrieval import DEFAULT_METHOD, METHODS, build_index, retrieve
from vector_store import cleanup_stale_sessions, session_store

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

METHOD_LABELS = {
    "rrf": "Hybrid — Reciprocal Rank Fusion",
    "weighted": "Hybrid — weighted scores",
    "bm25": "Keywords only (BM25)",
    "semantic": "Semantic only",
}

st.set_page_config(
    page_title="NovaBot — AI Book Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --bg-primary: #0f0f13; --bg-secondary: #16161d; --bg-card: #1c1c27;
    --accent-primary: #6c5ce7; --accent-secondary: #a29bfe;
    --accent-gradient: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 50%, #74b9ff 100%);
    --accent-glow: rgba(108, 92, 231, 0.3);
    --text-primary: #e8e6f0; --text-secondary: #9896a6; --text-muted: #6b6980;
    --border-color: rgba(108, 92, 231, 0.15);
    --success: #00cec9; --radius-sm: 8px; --radius-md: 12px; --radius-lg: 16px;
    --shadow-glow: 0 0 30px var(--accent-glow);
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg-primary) !important; color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
[data-testid="stHeader"] { background-color: transparent !important; }
.main .block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1100px !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-primary); border-radius: 3px; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13131a 0%, #0f0f16 100%) !important;
    border-right: 1px solid var(--border-color) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stMarkdown p {
    font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important;
}
.hero-container { text-align: center; padding: 3rem 1rem 2rem 1rem; margin-bottom: 1.5rem; }
.hero-icon { font-size: 4rem; display: inline-block; animation: float 3s ease-in-out infinite; margin-bottom: 0.8rem; }
@keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-12px); } }
.hero-title {
    font-size: 2.8rem; font-weight: 800; background: var(--accent-gradient);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: -1px; margin: 0; line-height: 1.15;
}
.hero-subtitle { color: var(--text-secondary); font-size: 1.05rem; font-weight: 400; margin-top: 0.6rem; }
.hero-divider { width: 60px; height: 3px; background: var(--accent-gradient); border-radius: 2px; margin: 1.5rem auto 0; }
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important; border: 2px dashed var(--border-color) !important;
    border-radius: var(--radius-lg) !important; padding: 1.5rem !important; transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-primary) !important; box-shadow: var(--shadow-glow) !important; }
[data-testid="stFileUploader"] label { font-size: 1rem !important; font-weight: 600 !important; color: var(--text-primary) !important; }
.status-card {
    background: linear-gradient(135deg, rgba(108,92,231,0.08), rgba(162,155,254,0.05));
    border: 1px solid var(--border-color); border-radius: var(--radius-lg);
    padding: 1.4rem 1.8rem; margin: 1.5rem 0; display: flex; align-items: center; gap: 1.2rem;
    animation: slideUp 0.5s ease-out;
}
@keyframes slideUp { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
.status-icon { font-size: 1.6rem; }
.status-text .main-text { font-weight: 600; font-size: 0.95rem; color: var(--text-primary); }
.status-text .sub-text { font-size: 0.82rem; color: var(--text-secondary); margin-top: 2px; }
.metrics-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0 2rem 0; }
.metric-card {
    background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-md);
    padding: 1.2rem; text-align: center; transition: all 0.3s ease; animation: fadeIn 0.6s ease-out;
}
.metric-card:hover { border-color: var(--accent-primary); transform: translateY(-2px); box-shadow: var(--shadow-glow); }
.metric-value {
    font-size: 1.6rem; font-weight: 700; background: var(--accent-gradient);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.metric-label { font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; font-weight: 500; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important; border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important; padding: 1rem 1.4rem !important;
    margin-bottom: 0.8rem !important; animation: slideUp 0.35s ease-out;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] code {
    color: var(--text-primary) !important; font-size: 0.92rem !important; line-height: 1.7 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { border-left: 3px solid var(--accent-primary) !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--success) !important;
    background: linear-gradient(135deg, rgba(0,206,201,0.04), var(--bg-card)) !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--bg-card) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important; border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea:focus { border-color: var(--accent-primary) !important; box-shadow: 0 0 0 2px var(--accent-glow) !important; }
[data-testid="stExpander"] { background: var(--bg-secondary) !important; border: 1px solid var(--border-color) !important; border-radius: var(--radius-md) !important; }
[data-testid="stExpander"] summary { color: var(--text-secondary) !important; font-weight: 500 !important; font-size: 0.88rem !important; }
[data-testid="stExpander"] summary:hover { color: var(--accent-secondary) !important; }
[data-testid="stAlert"] { background: var(--bg-card) !important; border: 1px solid var(--border-color) !important; border-radius: var(--radius-md) !important; color: var(--text-primary) !important; }
.sidebar-brand { text-align: center; padding: 1.5rem 0 1.2rem 0; border-bottom: 1px solid var(--border-color); margin-bottom: 1.5rem; }
.sidebar-brand-icon { font-size: 2.4rem; margin-bottom: 0.5rem; }
.sidebar-brand-name { font-size: 1.3rem; font-weight: 700; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.sidebar-brand-tag { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px; margin-top: 2px; }
.sidebar-section-title { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); margin: 1.5rem 0 0.8rem 0; }
.sidebar-info-card { background: rgba(108,92,231,0.06); border: 1px solid var(--border-color); border-radius: var(--radius-sm); padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.82rem; }
.sidebar-info-card .label { color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.sidebar-info-card .value { color: var(--accent-secondary); font-weight: 600; font-size: 0.85rem; }
.empty-state { text-align: center; padding: 4rem 2rem; animation: fadeIn 0.8s ease-out; }
.empty-state-icon { font-size: 4rem; margin-bottom: 1rem; opacity: 0.6; }
.empty-state-title { font-size: 1.4rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.6rem; }
.empty-state-desc { color: var(--text-secondary); font-size: 0.92rem; max-width: 420px; margin: 0 auto; line-height: 1.6; }
.powered-footer { text-align: center; padding: 2rem 0 1rem 0; color: var(--text-muted); font-size: 0.75rem; letter-spacing: 0.5px; }
.source-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-sm); padding: 1rem 1.2rem; margin-bottom: 0.6rem; transition: border-color 0.2s; }
.source-card:hover { border-color: var(--accent-primary); }
.source-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }
.source-badge { background: var(--accent-gradient); color: white; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; letter-spacing: 0.5px; }
.source-heading { font-weight: 600; font-size: 0.85rem; color: var(--text-primary); }
.source-meta { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.4rem; }
.source-preview { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.55; border-left: 2px solid var(--border-color); padding-left: 0.8rem; margin-top: 0.5rem; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🚀</div>
        <div class="sidebar-brand-name">NovaBot</div>
        <div class="sidebar-brand-tag">AI Book Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">⚙️ Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Sources to retrieve", min_value=2, max_value=10, value=5)
    method = st.selectbox("Retrieval method", METHODS, index=METHODS.index(DEFAULT_METHOD),
                          format_func=METHOD_LABELS.get)
    chunk_size = st.slider("Chunk size (words)", min_value=200, max_value=1000, value=500, step=50)
    chunk_overlap = st.slider("Chunk overlap (words)", min_value=0, max_value=300, value=100, step=25,
                              help="Capped at half the chunk size")
    use_reranker = st.toggle("🎯 Cross-encoder reranking", value=False, help="Uses a cross-encoder for more precise results (slower)")

    if st.button("🔄 Re-index current book", use_container_width=True,
                 help="Rebuilds this book's vectors from scratch"):
        st.session_state.force_reindex = True

    st.markdown('<div class="sidebar-section-title">🧠 Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-info-card"><div class="label">LLM</div><div class="value">{html.escape(llm.PRIMARY_MODEL)} (+ fallbacks)</div></div>
    <div class="sidebar-info-card"><div class="label">Provider</div><div class="value">Groq (Ultra-Fast)</div></div>
    <div class="sidebar-info-card"><div class="label">Retrieval</div><div class="value">{METHOD_LABELS[method]}</div></div>
    <div class="sidebar-info-card"><div class="label">Vector Store</div><div class="value">ChromaDB (one collection per session)</div></div>
    <div class="sidebar-info-card"><div class="label">Embeddings</div><div class="value">MiniLM-L6-v2</div></div>
    <div class="sidebar-info-card"><div class="label">Reranker</div><div class="value">MS-MARCO MiniLM (optional)</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="powered-footer">Built with Streamlit • Powered by GROQ</div>', unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-icon">🚀</div>
    <h1 class="hero-title">NovaBot</h1>
    <p class="hero-subtitle">
        Your AI-powered reading companion — ask anything about your book,<br>
        get instant, cited answers from the source material.
    </p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📄 Upload your PDF to get started", type="pdf",
    help="Upload any PDF book. NovaBot will index it and let you ask questions.")


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex


@st.cache_resource
def _cleanup_once():
    """Remove other sessions' expired vectors once per server process."""
    try:
        return cleanup_stale_sessions()
    except Exception as e:
        logging.getLogger(__name__).warning("Session cleanup skipped: %s", e)
        return 0


def load_book(pdf_bytes, chunk_size, chunk_overlap, force=False):
    """Build (or reuse) this session's index for the uploaded PDF."""
    overlap = safe_overlap(chunk_size, chunk_overlap)
    doc_id = document_id(pdf_bytes, chunk_size, overlap)
    previous = st.session_state.get("book")
    if previous and previous["doc_id"] == doc_id and not force:
        return previous

    with st.spinner("📖 Reading & indexing PDF — the first run also downloads the embedding model..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            path = tmp.name
        try:
            title = extract_book_title(path)
            pages = extract_text_by_page(path)
        finally:
            os.unlink(path)
        started = time.perf_counter()
        chunks = chunk_with_overlap(pages, chunk_size=chunk_size, overlap=overlap)
        if "store" not in st.session_state:
            st.session_state.store = session_store(st.session_state.session_id)
        index = build_index(chunks, doc_id, st.session_state.store, force=force)
        book = {"doc_id": doc_id, "title": title, "index": index, "seconds": time.perf_counter() - started}

    if not previous or previous["doc_id"].split("-")[0] != doc_id.split("-")[0]:
        st.session_state.messages = []  # a different book starts a fresh conversation
    st.session_state.book = book
    return book


# ── Main ─────────────────────────────────────────────────────────────────────
_cleanup_once()

if uploaded_file:
    client = llm.get_client()
    if client is None:
        st.error("⚠️ GROQ_API_KEY not set. Add it to a .env file or to Settings → Secrets.")
        st.stop()

    book = load_book(uploaded_file.getvalue(), chunk_size, chunk_overlap,
                     force=st.session_state.pop("force_reindex", False))
    index = book["index"]
    if not index.chunks:
        st.error("No readable text found in this PDF — it may be scanned images, which need OCR.")
        st.stop()

    sections = len({c["heading"] for c in index.chunks if not c["heading"].startswith("Pages ")})
    st.markdown(f"""
    <div class="status-card">
        <div class="status-icon">✅</div>
        <div class="status-text">
            <div class="main-text">"{html.escape(book['title'])}" indexed in {book['seconds']:.1f}s</div>
            <div class="sub-text">Hybrid search ready — vectors are stored in this session's own ChromaDB collection.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card"><div class="metric-value">{len(index.chunks)}</div><div class="metric-label">Chunks</div></div>
        <div class="metric-card"><div class="metric-value">{max(c['end_page'] for c in index.chunks) + 1}</div><div class="metric-label">Pages</div></div>
        <div class="metric-card"><div class="metric-value">{sections}</div><div class="metric-label">Sections found</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Retrieval scope (metadata filtering) ──────────────────────────────
    total_pages = max(c['end_page'] for c in index.chunks) + 1

    with st.expander("🔎 Retrieval scope", expanded=False):
        scope_pages = st.toggle("Restrict search to a page range", value=False,
                                help="Uses ChromaDB metadata filtering so retrieval only considers chunks in the selected pages")
        if scope_pages:
            col1, col2 = st.columns(2)
            from_page = col1.number_input("From page", min_value=1, max_value=total_pages, value=1)
            to_page = col2.number_input("To page", min_value=from_page, max_value=total_pages, value=total_pages)
            page_start, page_end = from_page - 1, to_page - 1
        else:
            page_start = page_end = None

    # ── Chat ─────────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about the book...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            history = st.session_state.messages[:-1]
            t_start = time.perf_counter()
            with st.spinner("🔍 Searching the book..."):
                search_query = llm.condense_question(client, query, history)
                chunks = retrieve(
                    index, search_query, top_k=top_k, method=method, use_reranker=use_reranker,
                    page_start=page_start, page_end=page_end,
                )
            t_retrieved = time.perf_counter()

            if chunks:
                messages = llm.build_messages(query, chunks, history, book["title"])
                response_text = st.write_stream(llm.stream_answer(client, messages))
            else:
                response_text = llm.NOT_FOUND
                st.markdown(response_text)
            t_answered = time.perf_counter()

            citations = llm.check_citations(response_text, len(chunks))
            if citations["invalid"]:
                st.warning(f"The answer cites source(s) {citations['invalid']} that weren't retrieved — "
                           "treat those parts with care.")
            elif chunks and not citations["cited"] and not citations["abstained"]:
                st.caption("⚠️ This answer has no source citations.")

            timing = f"⏱ search {t_retrieved - t_start:.2f}s · answer {t_answered - t_retrieved:.2f}s"
            if search_query != query:
                timing += f" · searched for: “{search_query}”"
            st.caption(timing)

            with st.expander("📚 View source passages"):
                for i, c in enumerate(chunks, 1):
                    keywords = html.escape(", ".join(c.get("keywords") or [])) or "—"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-badge">SOURCE {i}</span>
                            <span class="source-heading">{html.escape(c['heading'])}</span>
                        </div>
                        <div class="source-meta">📄 Pages {c['start_page']+1}–{c['end_page']+1} &nbsp;•&nbsp; 🏷️ {keywords}</div>
                        <div class="source-preview">{html.escape(c['text'][:400])}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📚</div>
        <div class="empty-state-title">No book loaded yet</div>
        <div class="empty-state-desc">
            Upload a PDF above to begin. NovaBot splits the book into passages, indexes
            them for keyword and semantic search, and answers your questions with
            citations to the pages they came from.
        </div>
    </div>
    """, unsafe_allow_html=True)
