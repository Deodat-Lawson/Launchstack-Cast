"""CLIP embed smoke: shape + L2 normalization."""
import os

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(
    os.environ.get("OPEN_CLIP_DOWNLOAD_DISABLED") == "1",
    reason="OPEN_CLIP_DOWNLOAD_DISABLED=1; CLIP weights cannot be fetched",
)


def test_encode_images_shape_and_norm() -> None:
    from drone_search.embed import encode_images

    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    out = encode_images([img])
    assert out.shape == (1, 512)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_encode_text_shape_and_norm() -> None:
    from drone_search.embed import encode_text

    out = encode_text(["a person in a red jacket"])
    assert out.shape == (1, 512)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
