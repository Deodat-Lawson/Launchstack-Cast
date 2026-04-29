"""Paths and small typed config."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Paths:
    data_dir: Path
    raw: Path
    features: Path
    splits: Path

    @classmethod
    def from_env(cls) -> Paths:
        data_dir = Path(os.environ.get("DRONE_SEARCH_DATA_DIR", "./data")).resolve()
        return cls(
            data_dir=data_dir,
            raw=data_dir / "raw",
            features=data_dir / "features",
            splits=data_dir / "splits",
        )


def resolve_device(explicit: str | None = None) -> str:
    """Return a torch device string. Priority: explicit > env > MPS > CUDA > CPU."""
    choice = (explicit or os.environ.get("DRONE_SEARCH_DEVICE", "")).strip().lower()
    if choice in {"cpu", "cuda", "mps"}:
        return choice
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
