import chromadb
import re
import unicodedata

# Persistent ChromaDB store — survives Streamlit restarts, no server needed.
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "novabot_chunks"

_client = None
_collection = None


def _sanitize(text: str) -> str:
    """Strip non-ASCII characters that trip ChromaDB's encoder."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def get_client() -> chromadb.PersistentClient:
    """Lazily open the persistent client once per process."""
    global _client
    if _client is None:
        print(f"[vector] Opening persistent ChromaDB at {CHROMA_PATH}")
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection():
    """Get (or create) the shared chunk collection. Cosine distance metric."""
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[vector] Using collection '{COLLECTION_NAME}' (count={_collection.count()})")
    return _collection


def is_indexed(doc_id: str) -> bool:
    """True if any chunks for this doc_id are already persisted."""
    res = get_collection().get(where={"doc_id": doc_id}, limit=1, include=["metadatas"])
    return len(res["ids"]) > 0


def delete_document(doc_id: str) -> None:
    """Remove every chunk belonging to a doc (used by re-index)."""
    get_collection().delete(where={"doc_id": doc_id})
    print(f"[vector] Deleted doc '{doc_id}' from ChromaDB.")


def index_document(doc_id: str, chunks: list, embeddings, force: bool = False) -> bool:
    """
    Persist chunk embeddings to ChromaDB.

    Skips (and returns False) when the doc is already indexed, unless
    `force` is True, in which case the old vectors are replaced.

    Args:
        doc_id: stable identifier for the uploaded PDF
        chunks: list of {heading, text, start_page, end_page, ...}
        embeddings: (n_chunks, dim) numpy array
        force: if True, delete existing vectors for doc_id before adding
    """
    col = get_collection()
    if not force and is_indexed(doc_id):
        print(f"[vector] Doc '{doc_id}' already indexed — skipping.")
        return False
    if force:
        delete_document(doc_id)

    ids, documents, metadatas = [], [], []
    for i, (chunk, _emb) in enumerate(zip(chunks, embeddings)):
        ids.append(f"{doc_id}:{i}")
        documents.append(_sanitize(chunk["text"]))
        metadatas.append({
            "doc_id": doc_id,
            "global_idx": i,
            "start_page": int(chunk["start_page"]),
            "end_page": int(chunk["end_page"]),
        })

    col.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    print(f"[vector] Indexed {len(ids)} chunks for doc '{doc_id}'.")
    return True


def query_semantic(query_embedding, n_results: int, where: dict = None) -> dict:
    """
    Query the store for the n_results nearest chunks.

    Args:
        query_embedding: (dim,) numpy array
        n_results: number of neighbors to return
        where: ChromaDB metadata filter (e.g. {"doc_id": ..., "page": ...})

    Returns:
        raw ChromaDB response dict with keys: ids, distances, metadatas
    """
    col = get_collection()
    total = col.count()
    if total == 0:
        return None
    return col.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(n_results, total),
        where=where,
        include=["metadatas", "distances"],
    )
