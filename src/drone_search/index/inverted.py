"""Tag posting lists + IDF weighting.

Maps to HW1 inverted-index analog and TF-IDF lecture material. Each Document
is "tokenized" into discrete attribute tags (clothing color, action, etc.);
posting lists allow Boolean filter + IDF-weighted ranked retrieval.
"""
from __future__ import annotations


def build_index(documents):
    """Build {tag -> set(doc_ids)} posting lists and per-tag IDF."""
    raise NotImplementedError("week 5 deliverable")


def search(index, query_tags, *, mode: str = "boolean"):
    """Boolean (AND/OR) or IDF-ranked retrieval over tags."""
    raise NotImplementedError("week 6 deliverable")
