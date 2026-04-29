"""Standing-query watcher — the "Web Agent" half of the course title.

Maps to proposal §3 Layer 4 alert agent and routing/filtering rubric concept.
Given a saved query and a live Document stream, fire alerts on threshold crossing.
"""
from __future__ import annotations


def watch(query, document_stream, *, threshold: float = 0.85):
    """Yield alerts whenever a streamed Document scores above `threshold`."""
    raise NotImplementedError("week 11 deliverable")
