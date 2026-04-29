"""YOLOv8 + ByteTrack person detector + tracker.

Maps to proposal §3 Layer 1 — Ingestion & Detection. Treats the detector as a
black-box "tokenizer" — the IR substance lives downstream.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image


@dataclass(frozen=True, slots=True)
class Detection:
    frame_id: int
    t: float
    bbox: tuple[int, int, int, int]    # x, y, w, h
    track_id: int                       # ByteTrack assignment, -1 if untracked
    det_conf: float


_PERSON_CLASS_ID = 0  # COCO class index for "person"


def extract_detections(
    video_path: str | Path,
    *,
    fps: float = 1.0,
    conf: float = 0.25,
    max_frames: int | None = None,
    model_name: str = "yolov8x.pt",
    tracker: str = "bytetrack.yaml",
) -> Iterator[tuple[Detection, Image.Image]]:
    """Yield (Detection, crop_image) for each person detection in `video_path`.

    Frames are sampled at `fps` (default 1 fps to keep CLIP cost tractable
    later, per proposal §7 risk mitigation). Only person-class detections
    (COCO class 0) are emitted.
    """
    from ultralytics import YOLO

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    stride = max(1, round(src_fps / fps))

    model = YOLO(model_name)

    yielded = 0
    results = model.track(
        source=str(video_path),
        tracker=tracker,
        stream=True,
        classes=[_PERSON_CLASS_ID],
        conf=conf,
        verbose=False,
        vid_stride=stride,
    )
    for stride_idx, r in enumerate(results):
        frame_idx = stride_idx * stride
        t = frame_idx / src_fps

        if r.boxes is None or len(r.boxes) == 0:
            continue

        frame_bgr = r.orig_img
        if frame_bgr is None:
            continue

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None

        for i in range(len(boxes)):
            x1, y1, x2, y2 = (int(v) for v in xyxy[i])
            w, h = x2 - x1, y2 - y1
            if w <= 1 or h <= 1:
                continue

            track_id = int(ids[i]) if ids is not None else -1
            det = Detection(
                frame_id=int(frame_idx),
                t=float(t),
                bbox=(x1, y1, w, h),
                track_id=track_id,
                det_conf=float(confs[i]),
            )
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop_rgb)
            yield det, crop

            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return
