"""FAISS HNSW dense index over CLIP embeddings.

Maps to proposal §3 Layer 3 "Dense index" — vector-space cosine retrieval.
HW2 analog: vector model + cosine similarity.
"""
from __future__ import annotations


def build_index(embeddings, **kwargs):
    """Build a FAISS HNSW index over (N, D) L2-normalized embeddings."""
    raise NotImplementedError("week 3 deliverable")


def search(index, queries, k: int = 10):
    """Return top-k (scores, ids) for each query."""
    raise NotImplementedError("week 3 deliverable")
