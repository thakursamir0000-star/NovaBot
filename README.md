<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Llama_3.3_70B-blueviolet" />
  <img src="https://img.shields.io/badge/Inference-GROQ-orange" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<h1 align="center">🚀 NovaBot — AI Book Assistant</h1>

<p align="center">
  <strong>A hierarchical RAG system that lets you upload any PDF book and ask natural-language questions with cited, source-grounded answers.</strong>
</p>

<p align="center">
  <em>Built with Streamlit · Powered by GROQ · Hybrid Retrieval · Cross-Encoder Reranking</em>
</p>

---

## ✨ Features

- **📄 Upload Any PDF** — Works with any book, not hardcoded to a single title
- **🔍 Hybrid Retrieval** — BM25 (35%) + Semantic Search (55%) + LDA Topic Boost (10%)
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
│  │ Pre-built│  │ NumPy MatMul  │  │  Word Overlap    │  │
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
│  Model: Llama 3.3 70B (via GROQ)                        │
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
│                │    │  (500w, 100 overlap)│    │  • Batch Embeddings │
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
├── tree_index.py       # Hierarchical index: embeddings + BM25 + LDA topics
├── retrieval.py        # Hybrid retrieval + optional cross-encoder reranking
├── llm.py              # GROQ LLM integration with conversation memory
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
└── README.md           # This file
```

| Module | Responsibility |
|---|---|
| `ingest.py` | Extracts text page-by-page, detects book title, chunks with configurable overlap |
| `tree_index.py` | Batch-encodes embeddings, pre-builds BM25, runs LDA topic modeling per chapter |
| `retrieval.py` | 3-signal hybrid scoring + lazy-loaded cross-encoder reranker |
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
git clone https://github.com/yourusername/novabot.git
cd novabot

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

---

## 🧠 Technical Highlights

### Why Hybrid Retrieval?
- **BM25** excels at exact keyword matching (e.g., "gradient descent")
- **Semantic search** catches paraphrases (e.g., "how to reduce loss" → gradient descent)
- **Topic boost** rewards chunks from the same topic cluster as the query
- Together, they outperform any single method

### Why Sliding-Window Chunking?
- Old approach: regex-based splitting at headings → broke across PDFs, lost context at boundaries
- New approach: fixed-size windows with overlap → no context loss, works with any document

### Why Pre-built Indices?
- BM25 index is built once at upload time, not reconstructed per query
- Embeddings are batch-encoded in a single forward pass (32 chunks at a time)
- Cosine similarity uses NumPy matrix multiplication instead of per-chunk loops

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit with custom CSS |
| **LLM** | Llama 3.3 70B via GROQ |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) |
| **Keyword Search** | BM25Okapi (rank-bm25) |
| **Topic Modeling** | Latent Dirichlet Allocation (Gensim) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **PDF Parsing** | PyPDF2 |
| **NLP** | NLTK (tokenization, stopwords) |

---


