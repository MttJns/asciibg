"""Turn a base image into the fixed cell field, once."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from .piece import Field

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def cover_crop(img: Image.Image, aspect: float) -> Image.Image:
    """Crop to `aspect` (width / height) about the centre, keeping as much as possible."""
    w, h = img.size
    if w / h > aspect:
        new_w = int(round(h * aspect))
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = int(round(w / aspect))
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def field_from_image(
    path: str,
    cols: int,
    rows: int,
    *,
    aspect: float = 16 / 9,
    clip: Tuple[float, float] = (2.0, 98.0),
    gamma: float = 1.0,
    saturation: float = 1.0,
    gain: float = 1.0,
    center_dim: float = 0.0,
) -> Field:
    """Quantise an image to a cols x rows field of brightness and colour.

    `clip` are the percentiles mapped to the ends of the ramp, so the image's own
    darkest and brightest parts set the range. `center_dim` fades the middle of the
    frame, where a speaker's head sits.
    """
    img = Image.open(path).convert("RGB")
    img = cover_crop(img, aspect)
    small = img.resize((cols, rows), Image.BOX)
    rgb = np.asarray(small, dtype=np.float32) / 255.0

    luma = rgb @ LUMA
    lo, hi = np.percentile(luma, clip)
    level = np.zeros_like(luma) if hi - lo < 1e-6 else (luma - lo) / (hi - lo)
    level = np.clip(level, 0.0, 1.0)
    if gamma != 1.0:
        level = level ** gamma

    if saturation != 1.0:
        grey = (rgb @ LUMA)[..., None]
        rgb = np.clip(grey + (rgb - grey) * saturation, 0.0, 1.0)
    if gain != 1.0:
        rgb = np.clip(rgb * gain, 0.0, 1.0)

    if center_dim > 0.0:
        level = level * (1.0 - center_dim * _center_mask(cols, rows))

    return Field(level=level.astype(np.float32), rgb=rgb.astype(np.float32))


def _center_mask(cols: int, rows: int) -> np.ndarray:
    """A soft ellipse over the centre-lower frame, where a head and torso sit."""
    x = (np.arange(cols, dtype=np.float32) + 0.5) / cols
    y = (np.arange(rows, dtype=np.float32) + 0.5) / rows
    dx = (x[None, :] - 0.5) / 0.30
    dy = (y[:, None] - 0.58) / 0.55
    d = np.sqrt(dx * dx + dy * dy)
    return np.clip(1.0 - d, 0.0, 1.0) ** 0.8
