"""Hybrid Boolean+ranked retrieval and Rocchio relevance feedback.

Maps to proposal §3 Layer 3 ranking score and §3 Layer 4 RF loop. HW2/HW3
analogs: vector-space scoring, Rocchio.

The hybrid score combines two retrieval signals:
- dense cosine similarity over CLIP embeddings (FAISS)
- IDF-weighted tag match over the inverted index

    score = alpha * cos(q, d) + beta * sum_t IDF(t) * match(t)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_search.document import Document
from drone_search.index import dense as dense_idx
from drone_search.index.inverted import InvertedIndex, score_tags


@dataclass(slots=True)
class Hit:
    doc_idx: int
    score: float
    dense_score: float
    tag_score: float


def hybrid_score(
    query_embedding: np.ndarray,
    query_tags: list[str],
    documents: list[Document],
    dense_index,
    inverted_index: InvertedIndex,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    k: int = 10,
    candidate_pool: int = 100,
) -> list[Hit]:
    """Combine dense cosine + IDF-weighted tag match.

    Pulls the top `candidate_pool` from the dense index, then re-ranks with
    the tag score. Returns the top-k hits sorted by combined score, descending.
    """
    pool = max(candidate_pool, k)
    scores, ids = dense_idx.search(dense_index, query_embedding, k=pool)
    cand_scores = scores[0]
    cand_ids = ids[0]

    hits: list[Hit] = []
    for cos, doc_idx in zip(cand_scores, cand_ids, strict=True):
        if doc_idx < 0 or doc_idx >= len(documents):
            continue
        doc = documents[int(doc_idx)]
        tag = score_tags(inverted_index, query_tags, doc.tags) if query_tags else 0.0
        combined = alpha * float(cos) + beta * tag
        hits.append(
            Hit(
                doc_idx=int(doc_idx),
                score=combined,
                dense_score=float(cos),
                tag_score=tag,
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


def rocchio(
    query_embedding: np.ndarray,
    relevant_embeddings: np.ndarray,
    irrelevant_embeddings: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> np.ndarray:
    """Rocchio query update: q' = a*q + b*mean(rel) - g*mean(irrel).

    Returns an L2-normalized embedding so the updated query stays comparable
    against the cosine-similarity dense index.
    """
    q = alpha * query_embedding.astype(np.float32)

    if relevant_embeddings is not None and len(relevant_embeddings) > 0:
        q = q + beta * np.mean(relevant_embeddings, axis=0).astype(np.float32)
    if irrelevant_embeddings is not None and len(irrelevant_embeddings) > 0:
        q = q - gamma * np.mean(irrelevant_embeddings, axis=0).astype(np.float32)

    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q.astype(np.float32)
