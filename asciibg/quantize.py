"""Turn a base image into the fixed cell field, once."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from .piece import Field

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def cover_crop(
    img: Image.Image,
    aspect: float,
    *,
    zoom: float = 1.0,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> Image.Image:
    """Crop to `aspect` (width / height), keeping as much of the image as possible.

    `zoom` above 1 crops tighter; `offset` then pans the window within what is left,
    each axis in -1..1. Together they move the subject off the centre of the frame,
    where a speaker's head sits.
    """
    w, h = img.size
    if w / h > aspect:
        cw, chh = h * aspect, float(h)
    else:
        cw, chh = float(w), w / aspect
    cw, chh = cw / max(zoom, 1e-6), chh / max(zoom, 1e-6)

    cx = w / 2.0 + offset[0] * max(0.0, (w - cw) / 2.0)
    cy = h / 2.0 + offset[1] * max(0.0, (h - chh) / 2.0)
    left = int(round(min(max(cx - cw / 2.0, 0.0), max(0.0, w - cw))))
    top = int(round(min(max(cy - chh / 2.0, 0.0), max(0.0, h - chh))))
    return img.crop((left, top, left + int(round(cw)), top + int(round(chh))))


def field_from_image(
    path: str,
    cols: int,
    rows: int,
    *,
    aspect: float = 16 / 9,
    zoom: float = 1.0,
    offset: Tuple[float, float] = (0.0, 0.0),
    clip: Tuple[float, float] = (2.0, 98.0),
    gamma: float = 1.0,
    saturation: float = 1.0,
    gain: float = 1.0,
    value_boost: float = 0.0,
    center_dim: float = 0.0,
) -> Field:
    """Quantise an image to a cols x rows field of brightness and colour.

    `clip` are the percentiles mapped to the ends of the ramp, so the image's own
    darkest and brightest parts set the range. Raise the low end for a picture that
    is mostly background, such as a subject on a night sky, or its subject is left
    crushed at the bottom of the ramp.

    `value_boost` lifts each cell's colour towards full value, keeping its hue.
    Without it a dim cell gets both a sparse character and a dark colour, which
    reads as doubly dark. `center_dim` fades the middle, where a speaker's head sits.
    """
    img = Image.open(path).convert("RGB")
    img = cover_crop(img, aspect, zoom=zoom, offset=offset)
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
    if value_boost > 0.0:
        peak = rgb.max(axis=-1, keepdims=True)
        full = rgb / np.maximum(peak, 1e-5)
        rgb = np.clip(rgb + (full - rgb) * value_boost, 0.0, 1.0)
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
