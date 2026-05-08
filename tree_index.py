import nltk
import numpy as np
from gensim import corpora, models
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words('english'))
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def sanitize_text(text) -> str:
    """Ensure text is a clean Python str suitable for tokenization."""
    if not isinstance(text, str):
        text = str(text)
    # Decode if bytes, handle encoding
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    # Remove null bytes and control characters (common in PDF extraction)
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t\r')
    text = text.replace('\x00', '').strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Ensure minimum length
    return text if text and len(text.split()) >= 3 else ""


def preprocess_text(text: str) -> list:
    """Lowercase, tokenize, remove stopwords and non-alpha tokens."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]


def safe_encode(embedder, texts):
    """
    Batch-encode a list of texts with proper error handling.
    Returns a 2D numpy array of shape (n_texts, embed_dim).
    """
    sanitized = []
    for text in texts:
        clean = sanitize_text(text)
        sanitized.append(clean if clean else "placeholder content")

    try:
        embeddings = embedder.encode(
            sanitized,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        return embeddings
    except Exception as e:
        print(f"[warn] Batch encoding failed: {e}. Using zero vectors.")
        return np.zeros((len(texts), 384))


def build_index(chunks: list, num_topics: int = 8) -> dict:
    """
    Builds a complete search index with multiple retrieval components.

    Architecture (3-level hierarchical + flat index):
      Level 1: Chapter grouping by heading prefix
      Level 2: Section chunks within each chapter
      Level 3: LDA topic distribution + embedding per chunk

    Additionally builds:
      - Pre-computed embedding matrix for fast numpy-based similarity
      - Pre-built BM25 index (not rebuilt per query)
      - Tokenized corpus for BM25

    Returns dict with keys:
      - tree: { chapter_name: [topic_node, ...] }
      - all_chunks: flat list of all topic nodes
      - embedding_matrix: np.ndarray (n_chunks, embed_dim)
      - bm25: pre-built BM25Okapi index
    """
    # Sanitize all chunk text and filter out empty ones
    for c in chunks:
        c['text'] = sanitize_text(c.get('text', ''))
    chunks = [c for c in chunks if c['text']]

    if not chunks:
        print("[tree] No valid chunks found after sanitization.")
        return {
            'tree': {},
            'all_chunks': [],
            'embedding_matrix': np.zeros((0, 384)),
            'bm25': None
        }

    # ── Step 1: Batch-encode ALL chunks at once (much faster) ────────────
    print(f"[tree] Encoding {len(chunks)} chunks...")
    all_texts = [c['text'] for c in chunks]
    embedding_matrix = safe_encode(embedder, all_texts)

    # ── Step 2: Pre-build BM25 index ────────────────────────────────────
    print("[tree] Building BM25 index...")
    tokenized_corpus = [text.lower().split() for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # ── Step 3: Group by chapter ────────────────────────────────────────
    chapters = {}
    chunk_indices = {}  # maps chapter -> list of global indices
    for idx, chunk in enumerate(chunks):
        chapter_key = chunk['heading'].split(':')[0].strip()
        chapters.setdefault(chapter_key, []).append(chunk)
        chunk_indices.setdefault(chapter_key, []).append(idx)

    print(f"[tree] Found {len(chapters)} chapters.")

    # ── Step 4: LDA topic modeling per chapter ──────────────────────────
    all_topic_nodes = []
    tree = {}

    for chapter, ch_chunks in chapters.items():
        ch_idx = chunk_indices[chapter]
        texts = [preprocess_text(c['text']) for c in ch_chunks]

        if len(texts) < 2:
            # Too few chunks for LDA — assign default topic
            topic_nodes = []
            for i, chunk in enumerate(ch_chunks):
                global_idx = ch_idx[i]
                node = {
                    "heading": chunk['heading'],
                    "text": chunk['text'],
                    "start_page": chunk['start_page'],
                    "end_page": chunk['end_page'],
                    "topic_id": 0,
                    "topic_words": [],
                    "embedding": embedding_matrix[global_idx].tolist(),
                    "_global_idx": global_idx,
                }
                topic_nodes.append(node)
                all_topic_nodes.append(node)
            tree[chapter] = topic_nodes
            continue

        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=1, no_above=0.9)
        corpus = [dictionary.doc2bow(t) for t in texts]

        n_topics = min(num_topics, len(texts))
        lda = models.LdaModel(
            corpus,
            num_topics=n_topics,
            id2word=dictionary,
            passes=15,
            random_state=42
        )

        topic_nodes = []
        for i, chunk in enumerate(ch_chunks):
            global_idx = ch_idx[i]
            bow = dictionary.doc2bow(preprocess_text(chunk['text']))
            topic_dist = dict(lda.get_document_topics(bow))
            dominant_topic = max(topic_dist, key=topic_dist.get) if topic_dist else 0
            topic_words = [w for w, _ in lda.show_topic(dominant_topic, topn=6)]

            node = {
                "heading": chunk['heading'],
                "text": chunk['text'],
                "start_page": chunk['start_page'],
                "end_page": chunk['end_page'],
                "topic_id": int(dominant_topic),
                "topic_words": topic_words,
                "embedding": embedding_matrix[global_idx].tolist(),
                "_global_idx": global_idx,
            }
            topic_nodes.append(node)
            all_topic_nodes.append(node)

        tree[chapter] = topic_nodes
        print(f"[tree] Chapter '{chapter}': {len(topic_nodes)} nodes, {n_topics} topics.")

    # CRITICAL: sort by _global_idx so all_chunks aligns with
    # embedding_matrix and BM25 index order (both use original chunk order)
    all_topic_nodes.sort(key=lambda x: x['_global_idx'])

    return {
        'tree': tree,
        'all_chunks': all_topic_nodes,
        'embedding_matrix': embedding_matrix,
        'bm25': bm25,
    }
