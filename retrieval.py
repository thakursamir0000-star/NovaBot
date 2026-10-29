"""Hybrid retrieval over one indexed book.

Two first-stage retrievers produce ranked lists:
  - BM25 over tokenized chunk text (exact terms, names, codes)
  - dense cosine similarity over sentence embeddings (paraphrases)
They are combined with Reciprocal Rank Fusion ("rrf", default) or a weighted
sum of min-max normalized scores ("weighted"); either retriever can also be
used alone. An optional cross-encoder then re-orders the candidates.
`evaluation/evaluate.py` compares all of these.
"""
import re
from collections import Counter
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from embeddings import embed

METHODS = ("bm25", "semantic", "weighted", "rrf")
DEFAULT_METHOD = "rrf"
RRF_K = 60
WEIGHTS = {"bm25": 0.4, "semantic": 0.6}
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same she should so some such than
that the their theirs them themselves then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your yours yourself yourselves
""".split())
_TOKEN = re.compile(r"[a-z0-9]+")

_reranker = None


def tokenize(text):
    """Lowercase word tokens without punctuation or stopwords ("descent?" -> "descent")."""
    return [t for t in _TOKEN.findall(str(text).lower()) if t not in _STOPWORDS]


def sanitize_text(text):
    """Drop control characters and collapse whitespace; '' if fewer than 3 words remain."""
    text = "".join(c for c in str(text) if ord(c) >= 32 or c in "\n\t\r")
    text = " ".join(text.replace("\x00", "").split())
    return text if len(text.split()) >= 3 else ""


@dataclass
class Index:
    doc_id: str
    chunks: list
    bm25: object
    store: object
    embedder: object = None


def build_index(chunks, doc_id, store, embedder=None, force=False):
    """Clean chunks, build BM25, and make sure the store holds this doc's vectors.

    Embedding is skipped when the store already has `doc_id` (e.g. the same PDF
    re-uploaded with the same chunk settings) unless `force` is set.
    """
    clean = []
    for chunk in chunks:
        text = sanitize_text(chunk.get("text", ""))
        if text:
            clean.append({**chunk, "text": text})

    tokenized = [tokenize(c["text"]) for c in clean]
    bm25 = BM25Okapi(tokenized) if any(tokenized) else None
    idf = bm25.idf if bm25 else {}
    for chunk, tokens in zip(clean, tokenized):
        # Display labels: most distinctive words (tf * idf), longer words first on ties.
        counts = Counter(t for t in tokens if not t.isdigit())
        chunk["keywords"] = sorted(counts, key=lambda t: (-counts[t] * idf.get(t, 0.0), -len(t), t))[:5]

    if clean:
        if force:
            store.delete(doc_id)
        if not store.has(doc_id):
            store.add(doc_id, clean, embed([c["text"] for c in clean], embedder))
    return Index(doc_id=doc_id, chunks=clean, bm25=bm25, store=store, embedder=embedder)


def _in_scope(chunk, page_start, page_end):
    return ((page_start is None or chunk["end_page"] >= page_start)
            and (page_end is None or chunk["start_page"] <= page_end))


def bm25_ranking(index, query, page_start=None, page_end=None):
    """[(chunk_index, bm25_score)] best first, only chunks that share a term with the query."""
    tokens = tokenize(query)
    if index.bm25 is None or not tokens:
        return []
    scores = index.bm25.get_scores(tokens)
    ranked = [(i, float(s)) for i, s in enumerate(scores)
              if s > 0 and _in_scope(index.chunks[i], page_start, page_end)]
    return sorted(ranked, key=lambda pair: -pair[1])


def semantic_ranking(index, query, n, page_start=None, page_end=None):
    """[(chunk_index, cosine_similarity)] best first, top-n from the vector store."""
    vector = embed([query], index.embedder)[0]
    return index.store.query(index.doc_id, vector, n, page_start, page_end)


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Sum 1 / (k + rank) across ranked lists. Uses ranks only, so the two retrievers'
    very different score scales don't need tuning."""
    scores = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: -pair[1])


def weighted_fusion(bm25_rank, semantic_rank, weights=WEIGHTS):
    """Weighted sum of min-max normalized scores."""
    def normalize(ranking):
        if not ranking:
            return {}
        values = [s for _, s in ranking]
        lo, hi = min(values), max(values)
        return {i: (s - lo) / (hi - lo) if hi > lo else 1.0 for i, s in ranking}

    b, s = normalize(bm25_rank), normalize(semantic_rank)
    combined = {i: weights["bm25"] * b.get(i, 0.0) + weights["semantic"] * s.get(i, 0.0) for i in b.keys() | s.keys()}
    return sorted(combined.items(), key=lambda pair: -pair[1])


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


def rerank(query, chunks, reranker=None):
    """Re-order chunks by cross-encoder relevance. The full chunk text is passed and
    the model truncates at 512 tokens (it used to see only the first 512 characters)."""
    reranker = reranker or get_reranker()
    scores = reranker.predict([(query, c["text"]) for c in chunks])
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)
    return sorted(chunks, key=lambda c: -c["rerank_score"])


def retrieve(index, query, top_k=5, method=DEFAULT_METHOD, use_reranker=False,
             page_start=None, page_end=None, candidates=None, reranker=None):
    """Return the top_k chunk dicts (with `chunk_id` and `score`) for `query`."""
    if method not in METHODS:
        raise ValueError(f"Unknown retrieval method {method!r}; choose from {METHODS}")
    if not index.chunks:
        return []

    n = candidates or max(top_k * 4, 20)
    bm25_rank = bm25_ranking(index, query, page_start, page_end)[:n] if method != "semantic" else []
    semantic_rank = semantic_ranking(index, query, n, page_start, page_end) if method != "bm25" else []

    if method == "bm25":
        fused = bm25_rank
    elif method == "semantic":
        fused = semantic_rank
    elif method == "weighted":
        fused = weighted_fusion(bm25_rank, semantic_rank)
    else:
        fused = reciprocal_rank_fusion([bm25_rank, semantic_rank])

    pool = fused[:n] if use_reranker else fused[:top_k]
    results = [{**index.chunks[i], "chunk_id": i, "score": round(score, 4)} for i, score in pool]
    if use_reranker and len(results) > 1:
        results = rerank(query, results, reranker)
    return results[:top_k]
