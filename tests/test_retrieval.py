import pytest

from embeddings import HashingEmbedder
from retrieval import (
    METHODS,
    bm25_ranking,
    build_index,
    reciprocal_rank_fusion,
    rerank,
    retrieve,
    tokenize,
    weighted_fusion,
)
from vector_store import InMemoryStore


def test_tokenize_strips_punctuation_and_stopwords():
    assert tokenize("What is gradient descent?") == ["gradient", "descent"]
    assert tokenize("151.625 MHz") == ["151", "625", "mhz"]


def test_bm25_finds_exact_terms(index):
    ranked = bm25_ranking(index, "radio frequency?")
    assert ranked[0][0] == 3


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_ranked_chunks(index, method):
    results = retrieve(index, "How high can drones fly?", top_k=3, method=method)
    assert results and results[0]["chunk_id"] == 1
    assert {"chunk_id", "score", "text", "keywords"} <= results[0].keys()


def test_unknown_method_is_rejected(index):
    with pytest.raises(ValueError):
        retrieve(index, "x", method="magic")


def test_page_scope_filters_both_retrievers(index):
    results = retrieve(index, "bears radio drones breakfast", top_k=5, page_start=3, page_end=4)
    assert {r["chunk_id"] for r in results} <= {3, 4}


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[(1, 9.0), (2, 5.0)], [(2, 0.9), (3, 0.8)]], k=60)
    assert fused[0][0] == 2
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_weighted_fusion_normalizes_each_list():
    fused = dict(weighted_fusion([(1, 30.0), (2, 10.0)], [(2, 0.9), (3, 0.1)], weights={"bm25": 0.5, "semantic": 0.5}))
    assert fused[1] == pytest.approx(0.5)
    assert fused[2] == pytest.approx(0.5)
    assert fused[3] == pytest.approx(0.0)


def test_reranker_sees_full_chunk_text(index):
    seen = []

    class FakeReranker:
        def predict(self, pairs):
            seen.extend(pairs)
            return [len(text) for _, text in pairs]

    long_text = "word " * 400
    chunks = [{"text": "short"}, {"text": long_text}]
    ranked = rerank("query", chunks, FakeReranker())
    assert ranked[0]["text"] == long_text
    assert seen[1][1] == long_text  # not cut to 512 characters


class CountingStore(InMemoryStore):
    def __init__(self):
        super().__init__()
        self.adds = 0

    def add(self, doc_id, chunks, vectors):
        self.adds += 1
        super().add(doc_id, chunks, vectors)


def test_existing_vectors_are_reused_unless_forced(chunks):
    store = CountingStore()
    build_index(chunks, "doc", store, embedder=HashingEmbedder())
    build_index(chunks, "doc", store, embedder=HashingEmbedder())
    assert store.adds == 1
    build_index(chunks, "doc", store, embedder=HashingEmbedder(), force=True)
    assert store.adds == 2


def test_empty_and_unreadable_chunks_are_dropped():
    index = build_index([{"heading": "h", "text": "\x00\x01", "start_page": 0, "end_page": 0}],
                        "doc", InMemoryStore(), embedder=HashingEmbedder())
    assert index.chunks == []
    assert retrieve(index, "anything") == []


def test_keywords_are_distinctive_terms(index):
    assert "sediment" in index.chunks[0]["keywords"]
    assert "the" not in index.chunks[0]["keywords"]
