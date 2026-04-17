"""
OpenCLIP visual embedding worker.

Input:
  { "inputs": { "image_paths": ["/path/to/frame1.jpg", ...] } }

Output:
  { "embeddings": [[0.01, -0.03, ...], ...] }
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.run import run


def visual_embed(request):
    inputs = request.get("inputs", {})
    image_paths = inputs.get("image_paths", [])

    if not image_paths:
        raise ValueError("inputs.image_paths is required and must be non-empty")

    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"image not found: {p}")

    import open_clip
    import torch
    from PIL import Image

    model_name = request.get("options", {}).get("model", "ViT-B-32")
    pretrained = request.get("options", {}).get("pretrained", "openai")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()

    images = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    batch = torch.stack(images)

    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)

    return {
        "embeddings": features.cpu().tolist(),
    }


if __name__ == "__main__":
    run(visual_embed)
