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
