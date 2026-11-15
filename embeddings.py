"""The sentence-embedding model, loaded once per process and shared by indexing and search."""
import os

import numpy as np

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", _DEFAULT_MODEL)
EMBEDDING_MODEL_REVISION = os.environ.get(
    "EMBEDDING_MODEL_REVISION",
    _DEFAULT_MODEL_REVISION if EMBEDDING_MODEL == _DEFAULT_MODEL else None,
)

_model = None


def get_embedder():
    """Lazily load the SentenceTransformer (imported here so tests don't need torch)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_MODEL_REVISION)
    return _model


class HashingEmbedder:
    """Dependency-free stand-in for tests and smoke runs: hashed bag of words.

    It only captures word overlap, NOT meaning, so numbers produced with it say
    nothing about real semantic search quality.
    """

    def __init__(self, dim=512):
        self.dim = dim

    def encode(self, texts, **kwargs):
        import zlib

        from retrieval import tokenize

        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        return vectors


def embed(texts, embedder=None):
    """Return L2-normalized float32 vectors, one row per text.

    `embedder` is anything with a SentenceTransformer-style `.encode()`; tests
    pass a small fake one.
    """
    embedder = embedder or get_embedder()
    vectors = np.asarray(
        embedder.encode(list(texts), batch_size=32, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)
