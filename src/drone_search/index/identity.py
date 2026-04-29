"""Per-track centroid index — HW2 centroid analog.

For each ByteTrack `track_id`, store the mean of detection embeddings as the
identity centroid. Supports "where was track #42 last seen" queries.
"""
from __future__ import annotations


def build_index(documents):
    """Group Documents by track_id; compute mean embedding centroid."""
    raise NotImplementedError("week 4 deliverable")


def lookup(index, track_id: int):
    """Return the centroid + last-seen metadata for a track."""
    raise NotImplementedError("week 4 deliverable")
