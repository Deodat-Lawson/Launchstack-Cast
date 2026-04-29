"""FAISS HNSW dense index over CLIP embeddings.

Maps to proposal §3 Layer 3 "Dense index" — vector-space cosine retrieval.
HW2 analog: vector model + cosine similarity.

Embeddings are assumed L2-normalized (open_clip output already is), so
inner-product over `IndexHNSWFlat(METRIC_INNER_PRODUCT)` == cosine similarity.
"""
from __future__ import annotations

import numpy as np


def build_index(embeddings: np.ndarray, *, m: int = 32, ef_construction: int = 200):
    """Build a FAISS HNSW index over (N, D) L2-normalized embeddings.

    `m` and `ef_construction` are HNSW graph-build parameters; defaults follow
    the FAISS docs' "good general values."
    """
    import faiss

    if embeddings.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {embeddings.shape}")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    n, d = embeddings.shape
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    if n > 0:
        index.add(embeddings)
    return index


def search(index, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, ids) of shape (Q, k) for each query embedding.

    Scores are cosine similarities in [-1, 1]; higher is more similar. Padded
    with -1 IDs and -inf scores when `index` has fewer than `k` items.
    """
    if queries.ndim == 1:
        queries = queries[np.newaxis, :]
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32)

    if index.ntotal == 0:
        return (
            np.full((queries.shape[0], k), -np.inf, dtype=np.float32),
            np.full((queries.shape[0], k), -1, dtype=np.int64),
        )

    scores, ids = index.search(queries, k)
    return scores, ids
