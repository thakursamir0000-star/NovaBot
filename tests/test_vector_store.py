import time

import numpy as np
import pytest

from vector_store import InMemoryStore

CHUNKS = [
    {"start_page": 0, "end_page": 0},
    {"start_page": 1, "end_page": 2},
    {"start_page": 3, "end_page": 3},
]
VECTORS = np.eye(3, dtype=np.float32)


def check_store(store):
    assert not store.has("doc")
    store.add("doc", CHUNKS, VECTORS)
    assert store.has("doc")

    results = store.query("doc", np.array([0.1, 1.0, 0.0], dtype=np.float32), n=2)
    assert [i for i, _ in results] == [1, 0]
    assert results[0][1] > results[1][1]

    scoped = store.query("doc", np.array([1.0, 0.0, 0.0], dtype=np.float32), n=3, page_start=2, page_end=3)
    assert {i for i, _ in scoped} == {1, 2}

    store.delete("doc")
    assert not store.has("doc")


def test_in_memory_store():
    check_store(InMemoryStore())


def test_chroma_store(tmp_path):
    pytest.importorskip("chromadb")
    from vector_store import ChromaStore

    check_store(ChromaStore("session_test", path=str(tmp_path)))


def test_sessions_cannot_see_each_others_documents(tmp_path):
    pytest.importorskip("chromadb")
    from vector_store import ChromaStore

    alice = ChromaStore("session_alice", path=str(tmp_path))
    bob = ChromaStore("session_bob", path=str(tmp_path))
    alice.add("doc", CHUNKS, VECTORS)
    assert not bob.has("doc")
    assert bob.query("doc", VECTORS[0], n=3) == []


def test_cleanup_removes_only_expired_sessions(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from vector_store import ChromaStore, cleanup_stale_sessions

    ChromaStore("session_fresh", path=str(tmp_path))
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("session_old", metadata={"hnsw:space": "cosine", "created_at": time.time() - 3 * 86400})
    client.create_collection("not_a_session", metadata={"hnsw:space": "cosine", "created_at": 0.0})

    assert cleanup_stale_sessions(path=str(tmp_path), max_age_hours=24) == 1
    names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    assert names == {"session_fresh", "not_a_session"}
