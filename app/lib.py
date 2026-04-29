"""Helpers shared across Streamlit tabs.

Keeps `streamlit_app.py` focused on layout and event wiring.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
from PIL import Image

from drone_search.config import Paths


@dataclass(slots=True)
class Upload:
    sha1: str
    original_filename: str
    n_docs: int
    n_frames: int
    ingested_at: str

    @property
    def video_path(self) -> Path:
        return Paths.from_env().data_dir / "uploads" / f"{self.sha1}.mp4"

    @property
    def docs_parquet(self) -> Path:
        return Paths.from_env().features / f"{self.sha1}.parquet"

    @property
    def frames_parquet(self) -> Path:
        return Paths.from_env().features / f"{self.sha1}.frames.parquet"


def manifest_path() -> Path:
    return Paths.from_env().data_dir / "uploads.json"


def load_manifest() -> dict[str, Upload]:
    p = manifest_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {sha: Upload(**meta) for sha, meta in raw.items()}


def save_manifest(uploads: dict[str, Upload]) -> None:
    p = manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        sha: {
            "sha1": u.sha1,
            "original_filename": u.original_filename,
            "n_docs": u.n_docs,
            "n_frames": u.n_frames,
            "ingested_at": u.ingested_at,
        }
        for sha, u in uploads.items()
    }
    p.write_text(json.dumps(raw, indent=2))


def store_upload(uploaded_bytes: bytes, original_filename: str) -> tuple[str, Path]:
    """Compute sha1, write the file under data/uploads/, return (sha1, path)."""
    sha = hashlib.sha1(uploaded_bytes).hexdigest()
    paths = Paths.from_env()
    upload_dir = paths.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{sha}.mp4"
    if not dest.exists():
        dest.write_bytes(uploaded_bytes)
    return sha, dest


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_ingest(video_path: Path, out_path: Path, *, caption: bool) -> subprocess.Popen[str]:
    """Spawn the ingest CLI as a subprocess. Caller streams stderr."""
    cmd = [
        "python",
        "-m",
        "drone_search",
        "ingest",
        "--video",
        str(video_path),
        "--out",
        str(out_path),
    ]
    if not caption:
        cmd.append("--no-caption")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


@lru_cache(maxsize=512)
def _frame_at(video_path_str: str, t_ms: int) -> Image.Image | None:
    """OpenCV seek to milliseconds, return PIL frame. Cached by (path, ms)."""
    cap = cv2.VideoCapture(video_path_str)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t_ms))
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def frame_at(video_path: Path, t_sec: float) -> Image.Image | None:
    return _frame_at(str(video_path), int(t_sec * 1000))


def crop_at(video_path: Path, t_sec: float, bbox: tuple[int, int, int, int]) -> Image.Image | None:
    img = frame_at(video_path, t_sec)
    if img is None:
        return None
    x, y, w, h = bbox
    return img.crop((x, y, x + w, y + h))
