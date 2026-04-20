"""
OpenCLIP visual embedding worker.

Input:
  { "inputs": { "image_paths": ["/path/to/frame1.jpg", ...] } }

Output:
  { "embeddings": [[0.01, -0.03, ...], ...] }

Runs as a long-lived daemon by default so the CLIP model is loaded once.
Pass --once for one-shot CLI execution. Model choice is fixed at init
(envs CAST_CLIP_MODEL, CAST_CLIP_PRETRAINED); per-request options are
honored only in one-shot mode.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.run import run, run_daemon


def _load_clip(model_name, pretrained):
    import open_clip
    import torch  # noqa: F401  (imported so daemon startup surfaces missing deps)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()
    return {"model": model, "preprocess": preprocess}


def init_clip():
    model_name = os.environ.get("CAST_CLIP_MODEL", "ViT-B-32")
    pretrained = os.environ.get("CAST_CLIP_PRETRAINED", "openai")
    return _load_clip(model_name, pretrained)


def visual_embed(request, state=None):
    inputs = request.get("inputs", {})
    image_paths = inputs.get("image_paths", [])

    if not image_paths:
        raise ValueError("inputs.image_paths is required and must be non-empty")

    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"image not found: {p}")

    import torch
    from PIL import Image

    if state is None:
        # One-shot path.
        model_name = request.get("options", {}).get("model", "ViT-B-32")
        pretrained = request.get("options", {}).get("pretrained", "openai")
        state = _load_clip(model_name, pretrained)

    model = state["model"]
    preprocess = state["preprocess"]

    images = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    batch = torch.stack(images)

    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)

    return {
        "embeddings": features.cpu().tolist(),
    }


if __name__ == "__main__":
    if "--once" in sys.argv:
        run(lambda req: visual_embed(req, None))
    else:
        run_daemon(visual_embed, init=init_clip)
