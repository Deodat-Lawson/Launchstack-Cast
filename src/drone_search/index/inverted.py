"""Tag posting lists + IDF weighting.

Maps to HW1 inverted-index analog and TF-IDF lecture material. Each Document
is "tokenized" into a set of attribute tags (clothing color, action, etc.);
posting lists allow Boolean filter and IDF-weighted ranked retrieval.

Tags are binary on a Document (a person either has the `red_jacket` tag or
they don't), so document TF is implicitly 1 — only IDF matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from drone_search.document import Document


@dataclass(slots=True)
class InvertedIndex:
    postings: dict[str, set[int]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    n_docs: int = 0


def build_index(documents: list[Document]) -> InvertedIndex:
    """Build {tag -> set(doc_idx)} posting lists and per-tag IDF.

    `doc_idx` is the document's position in the input list — callers are
    responsible for keeping that ordering stable across retrieval calls.
    """
    postings: dict[str, set[int]] = {}
    for i, d in enumerate(documents):
        for tag in d.tags:
            postings.setdefault(tag, set()).add(i)

    n = len(documents)
    idf: dict[str, float] = {}
    for tag, docs in postings.items():
        # Smoothed IDF (Manning et al. eq. 6.7) — never zero, never undefined.
        idf[tag] = math.log((n + 1) / (len(docs) + 1)) + 1.0

    return InvertedIndex(postings=postings, idf=idf, n_docs=n)


def boolean_search(index: InvertedIndex, query_tags: list[str], *, mode: str = "or") -> set[int]:
    """Return doc indices matching the tag set under AND ('and') or OR ('or')."""
    if not query_tags:
        return set()
    sets = [index.postings.get(t, set()) for t in query_tags]
    if mode == "and":
        return set.intersection(*sets) if sets else set()
    if mode == "or":
        return set.union(*sets) if sets else set()
    raise ValueError(f"unknown mode: {mode!r}")


def score_tags(index: InvertedIndex, query_tags: list[str], doc_tags: list[str]) -> float:
    """Sum of IDF over tags appearing in both query and document.

    This is the `sum_t IDF(t) * match(t)` term from the proposal's hybrid
    score formula. Unknown tags (not in the index) contribute 0.
    """
    matched = set(query_tags) & set(doc_tags)
    return sum(index.idf.get(t, 0.0) for t in matched)
