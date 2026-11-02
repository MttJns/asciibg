"""Procedural scenes: art computed per frame rather than quantised from an image.

A scene fills a frame's cells directly, the way ascii.rest's own scenes do. Every
scene is a function of the loop phase alone: anything that moves advances a whole
number of wraps per loop, and anything random is drawn from a table indexed by frame
number, so the last frame always joins back onto the first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .piece import Cells, Context, Meta

Stops = Sequence[Tuple[float, Tuple[float, float, float]]]

PALETTES: Dict[str, Stops] = {
    "ice": ((0.0, (0.02, 0.05, 0.18)), (0.5, (0.15, 0.55, 0.85)), (1.0, (0.85, 0.97, 1.0))),
    "amber": ((0.0, (0.10, 0.04, 0.0)), (0.5, (0.85, 0.42, 0.05)), (1.0, (1.0, 0.92, 0.70))),
    "green": ((0.0, (0.0, 0.08, 0.03)), (0.5, (0.10, 0.75, 0.25)), (1.0, (0.78, 1.0, 0.82))),
    "ember": ((0.0, (0.08, 0.0, 0.02)), (0.5, (0.80, 0.12, 0.08)), (1.0, (1.0, 0.85, 0.45))),
    "violet": ((0.0, (0.06, 0.0, 0.14)), (0.5, (0.60, 0.15, 0.80)), (1.0, (1.0, 0.80, 0.98))),
    "mono": ((0.0, (0.05, 0.05, 0.07)), (0.5, (0.55, 0.57, 0.62)), (1.0, (1.0, 1.0, 1.0))),
}


def gradient(t: np.ndarray, stops: Stops) -> np.ndarray:
    """Map values in 0..1 through a list of (position, colour) stops."""
    out = np.empty(t.shape + (3,), dtype=np.float32)
    out[...] = np.asarray(stops[0][1], dtype=np.float32)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        band = (t >= p0) & (t <= p1)
        if band.any():
            f = ((t[band] - p0) / max(p1 - p0, 1e-6))[:, None]
            out[band] = np.asarray(c0, np.float32) * (1 - f) + np.asarray(c1, np.float32) * f
    out[t > stops[-1][0]] = np.asarray(stops[-1][1], dtype=np.float32)
    return out


def coords(meta: Meta, aspect: float = 16 / 9) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates in 0..1, with x stretched so circles look round."""
    x = (np.arange(meta.cols, dtype=np.float32) + 0.5) / meta.cols
    y = (np.arange(meta.rows, dtype=np.float32) + 0.5) / meta.rows
    return (x[None, :] * aspect).astype(np.float32), np.repeat(
        y[:, None], 1, axis=1
    ).astype(np.float32)


def loop_noise(
    x: np.ndarray, y: np.ndarray, phase: float, *, octaves: int = 3, seed: int = 11, scale: float = 2.0
) -> np.ndarray:
    """Band-limited noise in 0..1 that repeats exactly once per loop.

    Built from sines whose time frequencies are whole numbers, which is what makes
    it seamless; true value noise would need a tileable lattice to manage the same.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
    weight = 0.0
    for octave in range(octaves):
        amp = 0.5 ** octave
        freq = scale * (2 ** octave)
        for _ in range(4):
            # Direction on a circle, not kx and ky drawn separately: independent draws
            # leave whole octaves nearly axis-aligned, which reads as banding.
            angle = float(rng.uniform(0, 2 * math.pi))
            kx, ky = freq * math.cos(angle), freq * math.sin(angle)
            spin = float(rng.integers(1, 3))
            offset = float(rng.uniform(0, 1))
            total += amp * np.sin(2 * math.pi * (kx * x + ky * y + spin * phase + offset))
            weight += amp
    return (total / max(weight, 1e-6)) * 0.5 + 0.5


def blank(meta: Meta) -> Cells:
    """An empty frame: nothing lit, no character overrides."""
    shape = (meta.rows, meta.cols)
    return Cells(
        level=np.zeros(shape, dtype=np.float32),
        rgb=np.zeros(shape + (3,), dtype=np.float32),
        char=np.full(shape, -1, dtype=np.int16),
    )


class Scene:
    """Base class. `setup` runs once, `render` runs per frame."""

    tags: Tuple[str, ...] = ()
    charset_extra: str = ""
    # Scenes with a smooth, continuous field want an isotropic ramp. Drawn with the
    # shade ramp they read as horizontal stripes, because `-`, `=` and `:` are all
    # horizontal strokes; blocks and dots have no direction.
    ramp_name: str = "shade"

    def setup(self, meta: Meta) -> None:
        self.meta = meta

    def render(self, ctx: Context) -> Cells:
        raise NotImplementedError

    def _stops(self, fallback: str) -> Stops:
        return PALETTES[getattr(self, "palette", "") or fallback]


@dataclass
class Starfield(Scene):
    """Layers of stars drifting past, nearer ones moving faster."""

    tags = ("star", "stars", "starfield", "space", "night", "sky", "galaxy", "cosmos")
    density: float = 0.055
    layers: float = 3.0
    speed: float = 1.0
    palette: str = "ice"

    def setup(self, meta: Meta) -> None:
        super().setup(meta)
        rng = np.random.default_rng(4242)
        stops = self._stops("ice")
        self._layers = []
        count = int(self.layers)
        for i in range(count):
            depth = (i + 1) / count
            n = max(1, int(self.density * meta.cols * meta.rows / count))
            # A whole number of screen widths per loop is what keeps the drift seamless.
            steps = max(1, int(round(self.speed * depth * 2)))
            shade = np.clip(rng.normal(0.45 + 0.45 * depth, 0.12, n), 0.12, 1.0).astype(np.float32)
            self._layers.append((
                rng.random(n).astype(np.float32),
                rng.integers(0, meta.rows, n),
                steps,
                shade,
                gradient(np.clip(rng.normal(0.55, 0.25, n), 0, 1).astype(np.float32), stops),
            ))

    def render(self, ctx: Context) -> Cells:
        cells = blank(self.meta)
        cols = self.meta.cols
        for x0, cy, steps, shade, colour in self._layers:
            cx = ((x0 + steps * ctx.phase) % 1.0 * cols).astype(np.int32) % cols
            cells.level[cy, cx] = shade
            cells.rgb[cy, cx] = colour
        return cells


@dataclass
class Rain(Scene):
    """Streaks falling at a slant."""

    tags = ("rain", "rainy", "storm", "shower", "drizzle", "downpour", "wet",
            "snow", "snowy", "snowfall", "blizzard", "falling", "ash")
    density: float = 0.55
    speed: float = 2.0
    length: float = 7.0
    palette: str = "ice"
    charset_extra = "|/"

    def setup(self, meta: Meta) -> None:
        super().setup(meta)
        rng = np.random.default_rng(90210)
        n = max(1, int(self.density * meta.cols))
        self._x = rng.integers(0, meta.cols, n)
        self._y = rng.random(n).astype(np.float32)
        self._steps = rng.integers(
            max(1, int(self.speed)), max(2, int(self.speed) + 3), n
        ).astype(np.float32)
        self._bright = rng.uniform(0.7, 1.0, n).astype(np.float32)
        self._glyph = meta.charset.index("|")
        self._stops_cache = self._stops("ice")

    def render(self, ctx: Context) -> Cells:
        cells = blank(self.meta)
        rows, cols = self.meta.rows, self.meta.cols
        head = ((self._y + self._steps * ctx.phase) % 1.0 * rows).astype(np.int32) % rows
        trail = max(1, int(self.length))
        for back in range(trail):
            fade = 1.0 - back / trail
            ys = (head - back) % rows
            cells.level[ys, self._x] = self._bright * fade
            cells.rgb[ys, self._x] = gradient(
                np.full(ys.shape, 0.25 + 0.6 * fade, dtype=np.float32), self._stops_cache
            )
            if back == 0:
                cells.char[ys, self._x] = self._glyph
        return cells


@dataclass
class Waves(Scene):
    """Interfering ripples, like light on water."""

    tags = ("wave", "waves", "water", "ocean", "sea", "ripple", "liquid", "flow", "calm")
    ramp_name = "block"
    scale: float = 3.0
    speed: float = 1.0
    contrast: float = 1.9
    palette: str = "ice"

    def setup(self, meta: Meta) -> None:
        super().setup(meta)
        self._x, self._y = coords(meta)
        self._stops_cache = self._stops("ice")
        rng = np.random.default_rng(777)
        self._terms = []
        for _ in range(5):
            angle = float(rng.uniform(0, 2 * math.pi))
            freq = float(rng.uniform(0.6, 1.0)) * self.scale
            self._terms.append((
                freq * math.cos(angle),
                freq * math.sin(angle),
                float(max(1, round(self.speed * rng.integers(1, 3)))),
                float(rng.uniform(0, 1)),
            ))

    def render(self, ctx: Context) -> Cells:
        cells = blank(self.meta)
        total = np.zeros((self.meta.rows, self.meta.cols), dtype=np.float32)
        for kx, ky, spin, offset in self._terms:
            total += np.sin(2 * math.pi * (kx * self._x + ky * self._y - spin * ctx.phase + offset))
        level = np.clip(
            (total / len(self._terms) * 0.5 + 0.5 - 0.5) * self.contrast + 0.5, 0.0, 1.0
        ).astype(np.float32)
        cells.level = level
        cells.rgb = gradient(level, self._stops_cache)
        return cells


@dataclass
class Nebula(Scene):
    """Slow clouds of looping noise."""

    tags = ("nebula", "cloud", "clouds", "fog", "mist", "smoke", "plasma", "aurora", "dream")
    ramp_name = "block"
    scale: float = 1.6
    speed: float = 1.0
    contrast: float = 2.4
    palette: str = "violet"

    def setup(self, meta: Meta) -> None:
        super().setup(meta)
        self._x, self._y = coords(meta)
        self._stops_cache = self._stops("violet")

    def render(self, ctx: Context) -> Cells:
        cells = blank(self.meta)
        n = loop_noise(self._x, self._y, ctx.phase * self.speed, octaves=3, scale=self.scale)
        level = np.clip((n - 0.5) * self.contrast + 0.5, 0.0, 1.0).astype(np.float32)
        cells.level = level
        cells.rgb = gradient(level, self._stops_cache)
        return cells


@dataclass
class Grid(Scene):
    """A perspective floor running away to a horizon."""

    tags = ("grid", "perspective", "floor", "retro", "synthwave", "tron", "road", "tunnel")
    speed: float = 1.0
    rows_apart: float = 0.42
    spokes: float = 24.0
    horizon: float = 0.40
    palette: str = "violet"

    def setup(self, meta: Meta) -> None:
        super().setup(meta)
        x = (np.arange(meta.cols, dtype=np.float32) + 0.5) / meta.cols
        y = (np.arange(meta.rows, dtype=np.float32) + 0.5) / meta.rows
        sx = (x[None, :] - 0.5) * 2.0
        sy = np.repeat(y[:, None], meta.cols, axis=1)

        self._below = sy > self.horizon
        depth = np.maximum(sy - self.horizon, 1e-3)
        # Distance along the floor, large near the horizon. Capped, because past the
        # cap the lines are closer together than a cell and only alias.
        self._w = (self.rows_apart / depth).astype(np.float32)
        # How much w changes across one cell row: the anti-aliasing width.
        self._dw = (self.rows_apart / depth ** 2 / meta.rows).astype(np.float32)
        self._spoke = (np.abs(sx) / depth).astype(np.float32)
        self._dspoke = np.maximum(
            np.abs(sx) / depth ** 2 / meta.rows, 1.0 / depth / meta.cols
        ).astype(np.float32)
        self._side = np.sign(sx).astype(np.float32)
        self._fade = np.clip((sy - self.horizon) * 3.0, 0.0, 1.0).astype(np.float32)
        self._glow = (np.clip(1.0 - np.abs(sy - self.horizon) * 14.0, 0.0, 1.0) * 0.55).astype(
            np.float32
        )
        self._stops_cache = self._stops("violet")

    @staticmethod
    def _lines(value: np.ndarray, width: np.ndarray) -> np.ndarray:
        frac = value % 1.0
        dist = np.minimum(frac, 1.0 - frac)
        return np.clip(1.0 - dist / np.clip(width, 1e-3, 0.5), 0.0, 1.0)

    def render(self, ctx: Context) -> Cells:
        cells = blank(self.meta)
        steps = max(1, int(round(self.speed * 2)))
        visible = self._w < 14.0

        toward = self._lines(self._w - steps * ctx.phase, self._dw) * visible
        across = self._lines(self._spoke * (self.spokes / 8.0), self._dspoke * (self.spokes / 8.0))

        level = np.maximum(toward, across) * self._fade * self._below
        level = np.clip(level + self._glow, 0.0, 1.0).astype(np.float32)
        cells.level = level
        cells.rgb = gradient(level, self._stops_cache)
        return cells


@dataclass
class Matrix(Scene):
    """Columns of characters falling, brightest at the head."""

    tags = ("matrix", "code", "terminal", "hacker", "digital", "glyph", "cyber", "data")
    density: float = 0.5
    speed: float = 1.0
    trail: float = 12.0
    palette: str = "green"
    charset_extra = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789<>/\\"

    def setup(self, meta: Meta) -> None:
        super().setup(meta)
        rng = np.random.default_rng(31337)
        n = max(1, int(self.density * meta.cols))
        self._cols = rng.choice(meta.cols, size=min(n, meta.cols), replace=False)
        self._start = rng.random(self._cols.size).astype(np.float32)
        self._steps = rng.integers(
            max(1, int(self.speed)), max(2, int(self.speed) + 3), self._cols.size
        ).astype(np.float32)
        first = meta.charset.index(self.charset_extra[0])
        # Characters come from a table indexed by frame, so they repeat with the loop.
        self._table = rng.integers(
            first, first + len(self.charset_extra),
            size=(meta.frame_count, meta.rows, self._cols.size), dtype=np.int16,
        )
        self._stops_cache = self._stops("green")
        self._rows_idx = np.arange(meta.rows, dtype=np.float32)[:, None]

    def render(self, ctx: Context) -> Cells:
        cells = blank(self.meta)
        rows = self.meta.rows
        head = (self._start + self._steps * ctx.phase) % 1.0 * rows
        dist = (self._rows_idx - head[None, :]) % rows
        trail = max(1.0, self.trail)
        lit = dist < trail
        shade = np.where(lit, 1.0 - dist / trail, 0.0).astype(np.float32)
        shade[dist < 1.0] = 1.0

        cells.level[:, self._cols] = shade
        cells.rgb[:, self._cols] = gradient(shade, self._stops_cache)
        glyphs = np.where(lit, self._table[ctx.frame], -1).astype(np.int16)
        cells.char[:, self._cols] = glyphs
        return cells


SCENES: Dict[str, type] = {
    "starfield": Starfield,
    "rain": Rain,
    "waves": Waves,
    "nebula": Nebula,
    "grid": Grid,
    "matrix": Matrix,
}


def parse_scene(spec: str) -> Scene:
    """Build a scene from a spec like `matrix:density=0.7,palette=amber`."""
    name, _, args = spec.partition(":")
    name = name.strip().lower()
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; choose from {', '.join(sorted(SCENES))}")
    kwargs: Dict[str, object] = {}
    for part in filter(None, (a.strip() for a in args.split(","))):
        key, _, value = part.partition("=")
        key = key.strip()
        kwargs[key] = value.strip() if key == "palette" else float(value)
    return SCENES[name](**kwargs)  # type: ignore[arg-type]
