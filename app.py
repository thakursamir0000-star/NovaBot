import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from ingest import extract_text_by_page, chunk_with_overlap, extract_book_title
from tree_index import build_index
from retrieval import retrieve
from llm import generate_answer

load_dotenv()

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
    num_topics = st.slider("Topics per chapter", min_value=3, max_value=15, value=8)
    chunk_size = st.slider("Chunk size (words)", min_value=200, max_value=1000, value=500, step=50)
    chunk_overlap = st.slider("Chunk overlap (words)", min_value=50, max_value=300, value=100, step=25)
    use_reranker = st.toggle("🎯 Cross-encoder reranking", value=False, help="Uses a cross-encoder for more precise results (slower)")

    st.markdown('<div class="sidebar-section-title">🧠 Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info-card"><div class="label">LLM</div><div class="value">Llama 3.3 70B</div></div>
    <div class="sidebar-info-card"><div class="label">Provider</div><div class="value">Groq (Ultra-Fast)</div></div>
    <div class="sidebar-info-card"><div class="label">Retrieval</div><div class="value">BM25 35% + Semantic 55% + Topic 10%</div></div>
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


@st.cache_data(show_spinner="📖 Reading & indexing PDF — this takes ~1–2 min on first run...")
def load_index(pdf_bytes, num_topics, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    book_title = extract_book_title(tmp_path)
    pages = extract_text_by_page(tmp_path)
    chunks = chunk_with_overlap(pages, chunk_size=chunk_size, overlap=chunk_overlap)
    index_data = build_index(chunks, num_topics=num_topics)
    os.unlink(tmp_path)
    return index_data, len(chunks), book_title


# ── Main ─────────────────────────────────────────────────────────────────────
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    index_data, chunk_count, book_title = load_index(pdf_bytes, num_topics, chunk_size, chunk_overlap)

    tree = index_data['tree']
    total_nodes = len(index_data['all_chunks'])
    chapters = len(tree)

    st.markdown(f"""
    <div class="status-card">
        <div class="status-icon">✅</div>
        <div class="status-text">
            <div class="main-text">"{book_title}" indexed successfully</div>
            <div class="sub-text">Overlapping chunks with hybrid retrieval ready.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card"><div class="metric-value">{chunk_count}</div><div class="metric-label">Chunks</div></div>
        <div class="metric-card"><div class="metric-value">{total_nodes}</div><div class="metric-label">Topic Nodes</div></div>
        <div class="metric-card"><div class="metric-value">{chapters}</div><div class="metric-label">Chapters</div></div>
    </div>
    """, unsafe_allow_html=True)

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
            with st.spinner("🔍 Searching the book..."):
                chunks = retrieve(query, index_data, top_k=top_k, use_reranker=use_reranker)

            stream = generate_answer(
                query, chunks,
                chat_history=st.session_state.messages[:-1],
                book_title=book_title
            )

            response_text = st.write_stream(
                (chunk.choices[0].delta.content or "")
                for chunk in stream
                if chunk.choices[0].delta.content
            )

            with st.expander("📚 View source passages"):
                for i, c in enumerate(chunks, 1):
                    topics = ", ".join(c['topic_words']) if c['topic_words'] else "general"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            <span class="source-badge">SOURCE {i}</span>
                            <span class="source-heading">{c['heading']}</span>
                        </div>
                        <div class="source-meta">📄 Pages {c['start_page']+1}–{c['end_page']+1} &nbsp;•&nbsp; 🏷️ {topics}</div>
                        <div class="source-preview">{c['text'][:400]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📚</div>
        <div class="empty-state-title">No book loaded yet</div>
        <div class="empty-state-desc">
            Upload a PDF above to begin. NovaBot will automatically index the book,
            extract chapters and topics, and build a searchable knowledge graph so you
            can ask natural-language questions and get cited answers instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)
