"""Vector stores for chunk embeddings.

- ChromaStore: persistent ChromaDB collection (used by the app). Each browser
  session gets its own collection, and collections older than a day are
  deleted, so one visitor's PDF is never searchable by another and the store
  doesn't grow forever.
- InMemoryStore: NumPy matrix with the same interface (tests, evaluation, and
  a fallback when ChromaDB isn't installed).

Both return [(chunk_index, cosine_similarity)] for the top-n chunks of one
document, optionally limited to a page range.
"""
import time

import numpy as np

CHROMA_PATH = "./chroma_db"
SESSION_PREFIX = "session_"
MAX_AGE_HOURS = 24
_ADD_BATCH = 1000  # ChromaDB rejects very large single add() calls


def _in_pages(start_page, end_page, page_start, page_end):
    return (page_start is None or end_page >= page_start) and (page_end is None or start_page <= page_end)


class InMemoryStore:
    def __init__(self):
        self._docs = {}

    def has(self, doc_id):
        return doc_id in self._docs

    def add(self, doc_id, chunks, vectors):
        pages = [(int(c["start_page"]), int(c["end_page"])) for c in chunks]
        self._docs[doc_id] = (np.asarray(vectors, dtype=np.float32), pages)

    def delete(self, doc_id):
        self._docs.pop(doc_id, None)

    def query(self, doc_id, vector, n, page_start=None, page_end=None):
        if doc_id not in self._docs:
            return []
        matrix, pages = self._docs[doc_id]
        sims = matrix @ np.asarray(vector, dtype=np.float32)
        results = []
        for i in np.argsort(-sims):
            if _in_pages(*pages[i], page_start, page_end):
                results.append((int(i), float(sims[i])))
                if len(results) >= n:
                    break
        return results


class ChromaStore:
    def __init__(self, collection_name, path=CHROMA_PATH):
        import chromadb

        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "created_at": time.time()},
        )

    def has(self, doc_id):
        found = self._collection.get(where={"doc_id": doc_id}, limit=1, include=["metadatas"])
        return len(found["ids"]) > 0

    def add(self, doc_id, chunks, vectors):
        # Only vectors + metadata are stored; chunk text stays in memory, which also
        # avoids the encoder errors raw PDF Unicode used to cause here.
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "chunk": i, "start_page": int(c["start_page"]), "end_page": int(c["end_page"])}
            for i, c in enumerate(chunks)
        ]
        vectors = np.asarray(vectors, dtype=np.float32)
        for s in range(0, len(ids), _ADD_BATCH):
            e = s + _ADD_BATCH
            self._collection.add(ids=ids[s:e], embeddings=vectors[s:e].tolist(), metadatas=metadatas[s:e])

    def delete(self, doc_id):
        self._collection.delete(where={"doc_id": doc_id})

    def query(self, doc_id, vector, n, page_start=None, page_end=None):
        conditions = [{"doc_id": {"$eq": doc_id}}]
        if page_start is not None:
            conditions.append({"end_page": {"$gte": int(page_start)}})
        if page_end is not None:
            conditions.append({"start_page": {"$lte": int(page_end)}})
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        total = self._collection.count()
        if total == 0:
            return []
        # Ask the HNSW index for the top-n only (it used to fetch every chunk).
        result = self._collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=min(n, total),
            where=where,
            include=["metadatas", "distances"],
        )
        return [
            (int(meta["chunk"]), max(0.0, 1.0 - float(dist)))
            for meta, dist in zip(result["metadatas"][0], result["distances"][0])
        ]


def cleanup_stale_sessions(path=CHROMA_PATH, max_age_hours=MAX_AGE_HOURS):
    """Delete per-session collections older than `max_age_hours`. Returns how many were removed."""
    import chromadb

    client = chromadb.PersistentClient(path=path)
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for item in client.list_collections():
        name = item if isinstance(item, str) else item.name  # API differs across chromadb versions
        if not name.startswith(SESSION_PREFIX):
            continue
        try:
            created = float((client.get_collection(name).metadata or {}).get("created_at", 0))
        except Exception:
            continue
        if created < cutoff:
            client.delete_collection(name)
            removed += 1
    return removed


def session_store(session_id):
    """Per-session ChromaDB store, or an in-memory store if ChromaDB is unavailable."""
    try:
        return ChromaStore(f"{SESSION_PREFIX}{session_id}")
    except ImportError:
        print("[vector] chromadb not installed — using in-memory vectors")
        return InMemoryStore()
