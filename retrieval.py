import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Models (loaded once) ─────────────────────────────────────────────────────
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Lazy-loaded cross-encoder for reranking
_reranker = None


def _get_reranker():
    """Lazy-load the cross-encoder reranker on first use."""
    global _reranker
    if _reranker is None:
        print("[retrieval] Loading cross-encoder reranker...")
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("[retrieval] Reranker loaded.")
    return _reranker


def retrieve(query: str, index_data: dict, top_k: int = 5, use_reranker: bool = False) -> list:
    """
    Hybrid retrieval with pre-built indices and optional cross-encoder reranking.

    Improvements over v1:
      1. Uses pre-built BM25 index (not rebuilt every query)
      2. Uses numpy matrix multiplication for batch cosine similarity (O(1) vs O(n) loop)
      3. Optional cross-encoder reranking of top candidates for precision
      4. Topic-aware boosting — chunks matching query topics get a bonus

    Args:
        query: user's question string
        index_data: dict from build_index() with keys: all_chunks, embedding_matrix, bm25
        top_k: number of results to return
        use_reranker: if True, applies cross-encoder reranking on top candidates

    Returns:
        list of top_k chunk dicts, ranked by relevance
    """
    all_chunks = index_data['all_chunks']
    embedding_matrix = index_data['embedding_matrix']
    bm25 = index_data['bm25']

    if not all_chunks:
        return []

    # ── BM25 scores (pre-built index, just score the query) ──────────────
    bm25_scores = np.array(bm25.get_scores(query.lower().split()))

    # Normalize BM25 to [0, 1]
    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    if bm25_max - bm25_min > 0:
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    # ── Semantic scores (fast numpy matmul instead of per-chunk loop) ────
    query_embedding = embedder.encode([query], convert_to_numpy=True)[0]

    # Cosine similarity via dot product of normalized vectors
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    matrix_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-10
    normalized_matrix = embedding_matrix / matrix_norms
    semantic_scores = normalized_matrix @ query_norm  # shape: (n_chunks,)

    # ── Topic boosting ───────────────────────────────────────────────────
    # Boost chunks whose LDA topic words overlap with query terms
    query_words = set(query.lower().split())
    topic_boost = np.zeros(len(all_chunks))
    for i, chunk in enumerate(all_chunks):
        topic_words = set(w.lower() for w in chunk.get('topic_words', []))
        overlap = len(query_words & topic_words)
        if overlap > 0:
            topic_boost[i] = min(overlap * 0.05, 0.15)  # max 15% boost

    # ── Hybrid score ─────────────────────────────────────────────────────
    hybrid_scores = 0.35 * bm25_norm + 0.55 * semantic_scores + 0.10 * topic_boost

    # ── Get top candidates ───────────────────────────────────────────────
    # Fetch more than top_k for reranking
    n_candidates = min(top_k * 3, len(all_chunks)) if use_reranker else top_k
    top_indices = np.argsort(hybrid_scores)[::-1][:n_candidates]
    candidates = [all_chunks[i] for i in top_indices]

    # ── Cross-encoder reranking (optional) ───────────────────────────────
    if use_reranker and len(candidates) > 1:
        reranker = _get_reranker()
        pairs = [[query, c['text'][:512]] for c in candidates]  # truncate for speed
        rerank_scores = reranker.predict(pairs)

        # Sort by reranker score
        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [c for c, _ in ranked[:top_k]]

    return candidates[:top_k]
