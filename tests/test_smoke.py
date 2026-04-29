"""Smoke test: every module imports."""
import importlib

MODULES = [
    "drone_search",
    "drone_search.config",
    "drone_search.document",
    "drone_search.ingest",
    "drone_search.embed",
    "drone_search.index",
    "drone_search.index.dense",
    "drone_search.index.inverted",
    "drone_search.index.identity",
    "drone_search.retrieve",
    "drone_search.cluster",
    "drone_search.triage",
    "drone_search.agent",
    "drone_search.eval",
    "drone_search.eval.metrics",
    "drone_search.llm",
]


def test_imports() -> None:
    for m in MODULES:
        importlib.import_module(m)
