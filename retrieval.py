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


def _build_where(doc_id: str = None, page_start: int = None, page_end: int = None) -> dict:
    """
    Build a ChromaDB metadata filter from the retrieval scope.

    Enables things like "only pages 10-20" or "only this document" to be
    pushed down into the vector store instead of post-filtering results.
    """
    conds = []
    if doc_id:
        conds.append({"doc_id": {"$eq": doc_id}})
    if page_start is not None:
        conds.append({"end_page": {"$gte": page_start}})
    if page_end is not None:
        conds.append({"start_page": {"$lte": page_end}})
    if len(conds) == 1:
        return conds[0]
    if len(conds) > 1:
        return {"$and": conds}
    return None


def _scope_mask(all_chunks: list, page_start: int = None, page_end: int = None) -> np.ndarray:
    """Boolean mask over all_chunks for chunks intersecting the page range."""
    mask = np.ones(len(all_chunks), dtype=bool)
    if page_start is not None:
        mask &= np.array([c['end_page'] >= page_start for c in all_chunks])
    if page_end is not None:
        mask &= np.array([c['start_page'] <= page_end for c in all_chunks])
    return mask


def _semantic_scores(query: str, all_chunks: list, collection, where: dict,
                     embedding_matrix: np.ndarray, doc_id: str = None) -> np.ndarray:
    """
    Semantic scores for every chunk, aligned with all_chunks order.

    Primary path: query the persistent ChromaDB collection and read the
    returned cosine distances. Fallback path: in-memory numpy matmul (only
    used if no vector store is wired up).
    """
    n = len(all_chunks)
    scores = np.zeros(n)

    if collection is not None:
        query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
        n_results = min(n, collection.count())
        if n_results <= 0:
            return scores
        result = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances"],
        )
        for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
            if doc_id is not None and meta.get("doc_id") != doc_id:
                continue
            idx = meta.get("global_idx")
            if isinstance(idx, int) and 0 <= idx < n:
                scores[idx] = max(0.0, 1.0 - dist)
        return scores

    # Fallback: cosine similarity via normalized dot product (no ChromaDB)
    if embedding_matrix is not None and len(embedding_matrix) == n:
        query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        matrix_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-10
        scores = (embedding_matrix / matrix_norms) @ query_norm
    return scores


def retrieve(query: str, index_data: dict, top_k: int = 5, use_reranker: bool = False,
             collection=None, doc_id: str = None,
             page_start: int = None, page_end: int = None) -> list:
    """
    Hybrid retrieval with a persistent ChromaDB semantic backend.

    Scoring is unchanged: BM25 (35%) + semantic (55%) + LDA topic boost (10%).
    Only the semantic signal now comes from ChromaDB queries (with optional
    metadata filtering) instead of an in-memory cosine-similarity loop.

    Args:
        query: user's question string
        index_data: dict from build_index() with keys: all_chunks, bm25, ...
        top_k: number of results to return
        use_reranker: if True, applies cross-encoder reranking on top candidates
        collection: ChromaDB collection for semantic retrieval
        doc_id: restrict retrieval to one document (metadata filter)
        page_start: 0-based lower page bound (metadata filter)
        page_end: 0-based upper page bound (metadata filter)

    Returns:
        list of top_k chunk dicts, ranked by relevance
    """
    all_chunks = index_data['all_chunks']
    bm25 = index_data['bm25']
    embedding_matrix = index_data.get('embedding_matrix')

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

    # ── Semantic scores (ChromaDB query instead of numpy matmul) ──────────
    where = _build_where(doc_id, page_start, page_end)
    semantic_scores = _semantic_scores(query, all_chunks, collection, where, embedding_matrix, doc_id)

    # ── Topic boosting ───────────────────────────────────────────────────
    # Boost chunks whose LDA topic words overlap with query terms
    query_words = set(query.lower().split())
    topic_boost = np.zeros(len(all_chunks))
    for i, chunk in enumerate(all_chunks):
        topic_words = set(w.lower() for w in chunk.get('topic_words', []))
        overlap = len(query_words & topic_words)
        if overlap > 0:
            topic_boost[i] = min(overlap * 0.05, 0.15)  # max 15% boost

    # ── Hybrid score (formula unchanged) ─────────────────────────────────
    hybrid_scores = 0.35 * bm25_norm + 0.55 * semantic_scores + 0.10 * topic_boost

    # Exclude chunks outside the requested page/document scope
    scope_mask = _scope_mask(all_chunks, page_start, page_end)
    hybrid_scores[~scope_mask] = -1e9

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
