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
    """Crop to `aspect` (width / height), keeping as much of the image as possible."""
    return frame_source(img, aspect, zoom=zoom, offset=offset)[0]


def frame_source(
    img: Image.Image,
    aspect: float,
    *,
    zoom: float = 1.0,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
    """Place the image in a window of `aspect`, returning it and where its pixels are.

    `zoom` above 1 crops tighter and the window stays inside the image. Below 1 the
    window is larger than the image, so the subject shrinks and the rest is padding;
    the returned box says which part of the window holds real pixels, in window
    coordinates. `offset` pans the window on each axis, in -1..1.
    """
    w, h = img.size
    if w / h > aspect:
        cw, chh = h * aspect, float(h)
    else:
        cw, chh = float(w), w / aspect
    cw, chh = cw / max(zoom, 1e-6), chh / max(zoom, 1e-6)

    slack_x, slack_y = (w - cw) / 2.0, (h - chh) / 2.0
    cx = w / 2.0 + offset[0] * abs(slack_x)
    cy = h / 2.0 + offset[1] * abs(slack_y)
    left, top = cx - cw / 2.0, cy - chh / 2.0
    if slack_x > 0:
        left = min(max(left, 0.0), w - cw)
    if slack_y > 0:
        top = min(max(top, 0.0), h - chh)

    left_i, top_i = int(round(left)), int(round(top))
    cw_i, chh_i = int(round(cw)), int(round(chh))
    window = img.crop((left_i, top_i, left_i + cw_i, top_i + chh_i))
    content = (
        max(0.0, -left_i / cw_i),
        max(0.0, -top_i / chh_i),
        min(1.0, (w - left_i) / cw_i),
        min(1.0, (h - top_i) / chh_i),
    )
    return window, content


def isolated_points(
    level: np.ndarray, floor: float, ceiling: float, isolation: float
) -> np.ndarray:
    """Cells that are dim, and brighter than their neighbours: stars, not a subject."""
    rows, cols = level.shape
    pad = np.pad(level, 1, mode="edge")
    local = sum(pad[y : y + rows, x : x + cols] for y in range(3) for x in range(3)) / 9.0
    return (level >= floor) & (level <= ceiling) & ((level - local) >= isolation)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean mask by `radius` cells in each direction."""
    rows, cols = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    span = 2 * radius + 1
    for dy in range(span):
        for dx in range(span):
            out |= padded[dy : dy + rows, dx : dx + cols]
    return out


def extend_sky(
    field: Field,
    padding: np.ndarray,
    *,
    subject: float = 0.4,
    margin: int = 2,
    seed: int = 7717,
) -> None:
    """Fill the padded cells by sampling the image's own background.

    Zooming out past the image's edges leaves padding that has to pass for more of
    the same sky. Copying only the stars is not enough: a photographed night sky is
    not black but a dim haze, so stars on true black read as a visible rectangle.
    Every padded cell therefore takes a whole random background cell, brightness and
    colour together, which reproduces the haze, the star density and their colour
    spread in one step.
    """
    real = ~padding
    if not padding.any() or not real.any():
        return
    sky = real & ~dilate(field.level > subject, margin)
    if not sky.any():
        return

    rng = np.random.default_rng(seed)
    source_level, source_rgb = field.level[sky], field.rgb[sky]
    pick = rng.integers(0, source_level.size, size=int(padding.sum()))
    field.level[padding] = source_level[pick]
    field.rgb[padding] = source_rgb[pick]


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
    sky_fill: bool = True,
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
    img, content = frame_source(img, aspect, zoom=zoom, offset=offset)
    small = img.resize((cols, rows), Image.BOX)
    rgb = np.asarray(small, dtype=np.float32) / 255.0

    padding = _padding_mask(content, cols, rows)
    luma = rgb @ LUMA
    # Percentiles come from real pixels only: padding is black, and counting it drags
    # the low end to zero, which silently disables the clip exactly when zoomed out.
    lo, hi = np.percentile(luma[~padding] if padding.any() else luma, clip)
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

    field = Field(level=level.astype(np.float32), rgb=rgb.astype(np.float32))

    if sky_fill:
        extend_sky(field, padding)
    if center_dim > 0.0:
        field.level = (field.level * (1.0 - center_dim * _center_mask(cols, rows))).astype(
            np.float32
        )
    return field


def _padding_mask(
    content: Tuple[float, float, float, float], cols: int, rows: int
) -> np.ndarray:
    """Cells whose centre falls outside the image's real pixels."""
    x0, y0, x1, y1 = content
    x = (np.arange(cols, dtype=np.float32) + 0.5) / cols
    y = (np.arange(rows, dtype=np.float32) + 0.5) / rows
    inside = ((x >= x0) & (x < x1))[None, :] & ((y >= y0) & (y < y1))[:, None]
    return ~inside


def _center_mask(cols: int, rows: int) -> np.ndarray:
    """A soft ellipse over the centre-lower frame, where a head and torso sit."""
    x = (np.arange(cols, dtype=np.float32) + 0.5) / cols
    y = (np.arange(rows, dtype=np.float32) + 0.5) / rows
    dx = (x[None, :] - 0.5) / 0.30
    dy = (y[:, None] - 0.58) / 0.55
    d = np.sqrt(dx * dx + dy * dy)
    return np.clip(1.0 - d, 0.0, 1.0) ** 0.8
