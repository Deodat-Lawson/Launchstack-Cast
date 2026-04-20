"""
InsightFace detection + ArcFace embedding worker.

Input:
  { "inputs": { "image_path": "/path/to/frame.jpg" } }

Output:
  { "detections": [{ "bbox": [x,y,w,h], "score": 0.99, "embedding": [0.01, ...] }] }

Runs as a long-lived daemon by default so the FaceAnalysis app (InsightFace
buffalo_l by default) is loaded once. Pass --once for one-shot CLI execution.
Model choice fixed at init (env CAST_FACES_MODEL); per-request options are
honored only in one-shot mode.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.run import run, run_daemon


def _load_faces(model_name):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def init_faces():
    model_name = os.environ.get("CAST_FACES_MODEL", "buffalo_l")
    return _load_faces(model_name)


def detect_faces(request, app=None):
    inputs = request.get("inputs", {})
    image_path = inputs.get("image_path")

    if not image_path:
        raise ValueError("inputs.image_path is required")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    import cv2
    import numpy as np

    if app is None:
        # One-shot path.
        model_name = request.get("options", {}).get("model", "buffalo_l")
        app = _load_faces(model_name)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"could not read image: {image_path}")

    faces = app.get(img)

    detections = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        detections.append(
            {
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(face.det_score),
                "embedding": (face.embedding / np.linalg.norm(face.embedding)).tolist(),
            }
        )

    return {"detections": detections}


if __name__ == "__main__":
    if "--once" in sys.argv:
        run(lambda req: detect_faces(req, None))
    else:
        run_daemon(detect_faces, init=init_faces)
