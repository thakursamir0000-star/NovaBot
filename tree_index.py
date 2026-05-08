import nltk
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = list(stopwords.words('english'))
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def sanitize_text(text) -> str:
    """Ensure text is a clean Python str suitable for tokenization."""
    if not isinstance(text, str):
        text = str(text)
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t\r')
    text = text.replace('\x00', '').strip()
    text = ' '.join(text.split())
    return text if text and len(text.split()) >= 3 else ""


def safe_encode(embedder, texts):
    """Batch-encode a list of texts with proper error handling."""
    sanitized = []
    for text in texts:
        clean = sanitize_text(text)
        sanitized.append(clean if clean else "placeholder content")
    try:
        embeddings = embedder.encode(
            sanitized, show_progress_bar=True,
            convert_to_numpy=True, batch_size=32
        )
        return embeddings
    except Exception as e:
        print(f"[warn] Batch encoding failed: {e}. Using zero vectors.")
        return np.zeros((len(texts), 384))


def build_index(chunks: list, num_topics: int = 8) -> dict:
    """
    Builds a complete search index with multiple retrieval components.
    Uses sklearn LDA instead of gensim for Python 3.14 compatibility.
    """
    for c in chunks:
        c['text'] = sanitize_text(c.get('text', ''))
    chunks = [c for c in chunks if c['text']]

    if not chunks:
        print("[tree] No valid chunks found after sanitization.")
        return {
            'tree': {}, 'all_chunks': [],
            'embedding_matrix': np.zeros((0, 384)), 'bm25': None
        }

    # ── Step 1: Batch-encode ALL chunks ──────────────────────────────────
    print(f"[tree] Encoding {len(chunks)} chunks...")
    all_texts = [c['text'] for c in chunks]
    embedding_matrix = safe_encode(embedder, all_texts)

    # ── Step 2: Pre-build BM25 index ─────────────────────────────────────
    print("[tree] Building BM25 index...")
    tokenized_corpus = [text.lower().split() for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # ── Step 3: Group by chapter ─────────────────────────────────────────
    chapters = {}
    chunk_indices = {}
    for idx, chunk in enumerate(chunks):
        chapter_key = chunk['heading'].split(':')[0].strip()
        chapters.setdefault(chapter_key, []).append(chunk)
        chunk_indices.setdefault(chapter_key, []).append(idx)

    print(f"[tree] Found {len(chapters)} chapters.")

    # ── Step 4: LDA topic modeling per chapter (sklearn) ─────────────────
    all_topic_nodes = []
    tree = {}

    for chapter, ch_chunks in chapters.items():
        ch_idx = chunk_indices[chapter]
        ch_texts = [c['text'] for c in ch_chunks]

        if len(ch_texts) < 2:
            topic_nodes = []
            for i, chunk in enumerate(ch_chunks):
                global_idx = ch_idx[i]
                node = {
                    "heading": chunk['heading'], "text": chunk['text'],
                    "start_page": chunk['start_page'], "end_page": chunk['end_page'],
                    "topic_id": 0, "topic_words": [],
                    "embedding": embedding_matrix[global_idx].tolist(),
                    "_global_idx": global_idx,
                }
                topic_nodes.append(node)
                all_topic_nodes.append(node)
            tree[chapter] = topic_nodes
            continue

        # Use sklearn CountVectorizer + LDA
        vectorizer = CountVectorizer(
            max_df=0.9, min_df=1, stop_words=STOP_WORDS, max_features=5000
        )
        try:
            dtm = vectorizer.fit_transform(ch_texts)
        except ValueError:
            # All stop words or empty — assign default
            topic_nodes = []
            for i, chunk in enumerate(ch_chunks):
                global_idx = ch_idx[i]
                node = {
                    "heading": chunk['heading'], "text": chunk['text'],
                    "start_page": chunk['start_page'], "end_page": chunk['end_page'],
                    "topic_id": 0, "topic_words": [],
                    "embedding": embedding_matrix[global_idx].tolist(),
                    "_global_idx": global_idx,
                }
                topic_nodes.append(node)
                all_topic_nodes.append(node)
            tree[chapter] = topic_nodes
            continue

        n_topics = min(num_topics, len(ch_texts))
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42,
            max_iter=15, learning_method='online'
        )
        doc_topics = lda.fit_transform(dtm)  # shape: (n_docs, n_topics)

        feature_names = vectorizer.get_feature_names_out()

        topic_nodes = []
        for i, chunk in enumerate(ch_chunks):
            global_idx = ch_idx[i]
            dominant_topic = int(np.argmax(doc_topics[i]))

            # Get top 6 words for the dominant topic
            topic_word_indices = lda.components_[dominant_topic].argsort()[-6:][::-1]
            topic_words = [feature_names[j] for j in topic_word_indices]

            node = {
                "heading": chunk['heading'], "text": chunk['text'],
                "start_page": chunk['start_page'], "end_page": chunk['end_page'],
                "topic_id": dominant_topic, "topic_words": topic_words,
                "embedding": embedding_matrix[global_idx].tolist(),
                "_global_idx": global_idx,
            }
            topic_nodes.append(node)
            all_topic_nodes.append(node)

        tree[chapter] = topic_nodes
        print(f"[tree] Chapter '{chapter}': {len(topic_nodes)} nodes, {n_topics} topics.")

    # CRITICAL: sort by _global_idx so all_chunks aligns with
    # embedding_matrix and BM25 index order
    all_topic_nodes.sort(key=lambda x: x['_global_idx'])

    return {
        'tree': tree,
        'all_chunks': all_topic_nodes,
        'embedding_matrix': embedding_matrix,
        'bm25': bm25,
    }
