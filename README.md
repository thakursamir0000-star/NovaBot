<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-GPT--OSS_120B-blueviolet" />
  <img src="https://img.shields.io/badge/Inference-GROQ-orange" />
  <img src="https://img.shields.io/badge/Vector_Store-ChromaDB-00cec9" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<h1 align="center">🚀 NovaBot — AI Book Assistant</h1>

<p align="center">
  <strong>A hierarchical RAG system that lets you upload any PDF book and ask natural-language questions with cited, source-grounded answers.</strong>
</p>

<p align="center">
  <em>Built with Streamlit · Powered by GROQ · Hybrid Retrieval · ChromaDB Vector Store · Cross-Encoder Reranking</em>
</p>

<p align="center">
  <a href="https://novabot-hmfpdexedmytfg6ikpju3p.streamlit.app/">
    <img src="https://img.shields.io/badge/🔴_Live_Demo-NovaBot-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" />
  </a>
</p>

---

## ✨ Features

- **📄 Upload Any PDF** — Works with any book, not hardcoded to a single title
- **🔍 Hybrid Retrieval** — BM25 (35%) + Semantic Search (55%) + LDA Topic Boost (10%)
- **💾 Persistent Vector Store** — Embeddings live in ChromaDB (local, zero-server), surviving restarts without re-embedding
- **🎯 Metadata Filtering** — Scope retrieval to a page range or document with ChromaDB `where` filters
- **🌳 Hierarchical Tree Index** — 3-level architecture: Chapters → Sections → Topic Nodes
- **🎯 Cross-Encoder Reranking** — Optional MS-MARCO reranker for precision (toggle in sidebar)
- **💬 Conversation Memory** — Follow-up questions use chat history for context
- **📑 Source Citations** — Every answer includes page numbers and source passages
- **⚡ Streaming Responses** — Real-time token streaming via GROQ's ultra-fast inference
- **🎨 Premium Dark UI** — Custom-themed Streamlit interface with animations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      USER QUERY                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               HYBRID RETRIEVAL ENGINE                   │
│                                                         │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │  BM25    │  │   Semantic    │  │   LDA Topic      │  │
│  │  (35%)   │  │   (55%)       │  │   Boost (10%)    │  │
│  │ Pre-built│  │  ChromaDB     │  │  Word Overlap    │  │
│  │          │  │  cosine query │  │                  │  │
│  └────┬─────┘  └──────┬────────┘  └───────┬──────────┘  │
│       └───────────┬────┘───────────────────┘             │
│                   ▼                                      │
│         Combined Hybrid Score                            │
│                   │                                      │
│                   ▼                                      │
│     ┌──────────────────────────┐                         │
│     │  Cross-Encoder Reranker  │  (Optional)             │
│     │  MS-MARCO MiniLM-L6     │                          │
│     └────────────┬─────────────┘                         │
└──────────────────┼──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    LLM GENERATION                       │
│                                                         │
│  Model: GPT-OSS 120B (via GROQ)                          │
│  Context: Top-K Retrieved Passages                      │
│  Memory: Last 6 conversation turns                      │
│  Output: Streaming response with [Source N] citations   │
└─────────────────────────────────────────────────────────┘
```

### Indexing Pipeline

```
PDF Upload
    │
    ▼
┌────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│  PDF Ingestion │───▶│  Sliding-Window    │───▶│  Tree Index Builder │
│  (PyPDF2)      │    │  Chunking          │    │                     │
│                │    │  (500w, 100 overlap)│    │  • ChromaDB vectors │
│  • Page-by-page│    │                    │    │  • Pre-built BM25   │
│  • Title detect│    │  • Any PDF format  │    │  • LDA per Chapter  │
└────────────────┘    └────────────────────┘    └─────────────────────┘
```

---

## 📁 Project Structure

```
novabot/
├── app.py              # Streamlit UI + main application logic
├── ingest.py           # PDF text extraction + sliding-window chunking
├── tree_index.py       # Hierarchical index: BM25 + LDA topics + ChromaDB population
├── vector_store.py     # Persistent ChromaDB client + chunk indexing/querying
├── retrieval.py        # Hybrid retrieval + metadata filtering + optional reranking
├── llm.py              # GROQ LLM integration with conversation memory
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
└── README.md           # This file
```

| Module | Responsibility |
|---|---|
| `ingest.py` | Extracts text page-by-page, detects book title, chunks with configurable overlap |
| `tree_index.py` | Pre-builds BM25, runs LDA topic modeling per chapter, persists embeddings to ChromaDB |
| `vector_store.py` | Persistent ChromaDB client (cosine), chunk add/delete/query with metadata filters |
| `retrieval.py` | 3-signal hybrid scoring (semantic signal from ChromaDB) + lazy-loaded reranker |
| `llm.py` | Dynamic system prompt, conversation memory, streaming GROQ responses |
| `app.py` | Premium Streamlit UI with sidebar controls and source citation display |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [GROQ API key](https://console.groq.com/) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/thakursamir0000-star/NovaBot.git
cd NovaBot

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser, upload a PDF, and start asking questions!

> A local `./chroma_db/` directory is created automatically on first run (it is git-ignored) — no separate database server is required. If you ever change the vector store, just click **🔄 Re-index current book** in the sidebar.

---

## ⚙️ Configurable Parameters

All tunable from the sidebar:

| Parameter | Default | Range | Description |
|---|---|---|---|
| Sources to retrieve | 5 | 2–10 | Number of passages returned per query |
| Topics per chapter | 8 | 3–15 | LDA topics extracted during indexing |
| Chunk size (words) | 500 | 200–1000 | Words per chunk in sliding window |
| Chunk overlap (words) | 100 | 50–300 | Overlapping words between chunks |
| Cross-encoder reranking | Off | On/Off | MS-MARCO reranker for higher precision |
| Page-range scope | Off | On/Off | Restrict retrieval to a page range via ChromaDB metadata filtering |

---

## 🧠 Technical Highlights

### Why Hybrid Retrieval?
- **BM25** excels at exact keyword matching (e.g., "gradient descent")
- **Semantic search** (ChromaDB cosine query) catches paraphrases (e.g., "how to reduce loss" → gradient descent)
- **Topic boost** rewards chunks from the same topic cluster as the query
- Together, they outperform any single method

### Why Sliding-Window Chunking?
- Old approach: regex-based splitting at headings → broke across PDFs, lost context at boundaries
- New approach: fixed-size windows with overlap → no context loss, works with any document

### Why Pre-built Indices?
- BM25 index is built once at upload time, not reconstructed per query
- Embeddings are batch-encoded in a single forward pass (32 chunks at a time) and persisted to ChromaDB
- Semantic queries hit the persistent HNSW index (cosine) instead of scanning every vector in a Python loop

### Why ChromaDB?
- **What it replaced**: the old semantic signal re-computed cosine similarity in-memory over a NumPy matrix on every query — nothing persisted between sessions, and there was no way to scope results.
- **Persistent**: embeddings survive Streamlit restarts — re-uploading a book skips the expensive re-embedding pass (checked by `doc_id`)
- **Metadata filtering**: page numbers and doc IDs are stored as metadata, so scope filters ("only pages 10–20") are pushed into the vector store query instead of post-filtering results
- **Scalable**: HNSW approximate nearest-neighbor search beats a full linear scan once a book grows large
- **Zero infrastructure**: runs locally as a `PersistentClient` — no separate server to deploy
- **Cross-document ready**: every chunk is tagged with a `doc_id`, enabling multi-book search later

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit with custom CSS |
| **LLM** | GPT-OSS 120B via GROQ |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) |
| **Vector Store** | ChromaDB (persistent, HNSW, cosine distance) |
| **Keyword Search** | BM25Okapi (rank-bm25) |
| **Topic Modeling** | Latent Dirichlet Allocation (scikit-learn) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **PDF Parsing** | PyPDF2 |
| **NLP** | NLTK (tokenization, stopwords) |

---


