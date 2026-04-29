"""CLIP image + text encoder.

Maps to proposal §3 Layer 2 (CLIP image embedding) and the cross-modal text
query path (§3.4 query type 2). Single shared model, lazy-loaded on first use.

Ported from the original Cast worker in python/workers/visual_embed.py
(now archived on the cast-archive branch). Inlined as a regular module here
because the Streamlit app is single-process — no subprocess isolation needed.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _load(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, preprocess, tokenizer, device


def encode_images(
    images: Sequence[Image.Image],
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 32,
) -> np.ndarray:
    """L2-normalized image embeddings, shape (N, 512)."""
    if not images:
        return np.zeros((0, 512), dtype=np.float32)

    import torch

    model, preprocess, _, device = _load(model_name, pretrained)
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            tensors = torch.stack([preprocess(img.convert("RGB")) for img in batch]).to(device)
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def encode_text(
    texts: Sequence[str],
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> np.ndarray:
    """L2-normalized text embeddings, shape (N, 512)."""
    if not texts:
        return np.zeros((0, 512), dtype=np.float32)

    import torch

    model, _, tokenizer, device = _load(model_name, pretrained)
    tokens = tokenizer(list(texts)).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)
