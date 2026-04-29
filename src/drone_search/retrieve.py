"""Hybrid Boolean+ranked retrieval and Rocchio relevance feedback.

Maps to proposal §3 Layer 3 ranking score and §3 Layer 4 RF loop. HW2/HW3
analogs: vector-space scoring, Rocchio.
"""
from __future__ import annotations


def hybrid_score(
    query_embedding,
    query_tags,
    dense_index,
    inverted_index,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    k: int = 10,
):
    """Combine dense cosine + IDF-weighted tag match: score = a*cos + b*sum(IDF*match)."""
    raise NotImplementedError("week 6 deliverable")


def rocchio(
    query_embedding,
    relevant_embeddings,
    irrelevant_embeddings,
    *,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
):
    """Rocchio query update: q' = a*q + b*mean(rel) - g*mean(irrel)."""
    raise NotImplementedError("week 8 deliverable")
