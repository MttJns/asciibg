"""Motion layers applied over a fixed cell field.

The field is quantised once; an effect only changes a frame's brightness, colour
and the occasional character. Every effect is a function of the loop phase alone,
built from periodic terms or from tables indexed by frame number, so the last
frame always joins back onto the first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .piece import Cells, Context


class Effect:
    """A motion layer. `apply` mutates one frame's cells in place."""

    def apply(self, cells: Cells, ctx: Context) -> None:
        raise NotImplementedError


def _brighten(cells: Cells, boost: np.ndarray, tint: float) -> None:
    cells.level = np.clip(cells.level + boost, 0.0, 1.0)
    if tint > 0.0:
        t = np.clip(boost * tint, 0.0, 1.0)[..., None]
        cells.rgb = np.clip(cells.rgb + (1.0 - cells.rgb) * t, 0.0, 1.0)


def _repeats(ctx: Context, period: float) -> float:
    """Whole number of times a sub-animation fits in the loop, so it stays seamless."""
    if period <= 0:
        return 1.0
    return max(1.0, round(ctx.meta.loop / period))


@dataclass
class Glint(Effect):
    """A bright band sweeping across the art, as ascii.rest's glint does."""

    period: float = 5.0
    sweep: float = 0.45
    width: float = 0.14
    strength: float = 0.55
    tint: float = 0.8
    angle: float = 20.0

    def apply(self, cells: Cells, ctx: Context) -> None:
        rows, cols = cells.level.shape
        local = (ctx.phase * _repeats(ctx, self.period)) % 1.0
        if local >= self.sweep:
            return
        travel = local / self.sweep

        rad = math.radians(self.angle)
        x = (np.arange(cols, dtype=np.float32) + 0.5) / cols
        y = (np.arange(rows, dtype=np.float32) + 0.5) / rows
        d = x[None, :] * math.cos(rad) + y[:, None] * math.sin(rad)
        span = abs(math.cos(rad)) + abs(math.sin(rad))
        d = d / span

        centre = -self.width + travel * (1.0 + 2.0 * self.width)
        boost = self.strength * np.exp(-((d - centre) / self.width) ** 2 * 4.0)
        _brighten(cells, boost.astype(np.float32), self.tint)


@dataclass
class Scan(Effect):
    """A line running down the art, scrambling the characters it passes."""

    period: float = 7.0
    thickness: float = 1.8
    strength: float = 0.35
    tint: float = 0.5
    scramble: float = 0.7

    def __post_init__(self) -> None:
        self._table: Dict[int, np.ndarray] = {}

    def _noise(self, ctx: Context, charset_size: int) -> np.ndarray:
        """A per-frame table of random characters, indexed by frame so it loops."""
        key = id(ctx.meta)
        if key not in self._table:
            rng = np.random.default_rng(20240617)
            rows, cols = ctx.meta.rows, ctx.meta.cols
            self._table[key] = rng.integers(
                1, charset_size, size=(ctx.meta.frame_count, rows, cols), dtype=np.int16
            )
        return self._table[key][ctx.frame]

    def apply(self, cells: Cells, ctx: Context) -> None:
        rows, cols = cells.level.shape
        local = (ctx.phase * _repeats(ctx, self.period)) % 1.0

        th = self.thickness
        centre = -th + local * (rows + 2.0 * th)
        y = np.arange(rows, dtype=np.float32) + 0.5
        falloff = np.clip(1.0 - np.abs(y - centre) / th, 0.0, 1.0)[:, None]
        if not falloff.any():
            return

        boost = (self.strength * falloff * np.ones((1, cols), dtype=np.float32)).astype(np.float32)
        _brighten(cells, boost, self.tint)

        if self.scramble > 0.0:
            charset_size = len(ctx.meta.charset)
            rng = np.random.default_rng(1013904223 + ctx.frame)
            hit = (falloff > 0.0) & (rng.random((rows, cols)) < self.scramble * falloff)
            if hit.any():
                cells.char = np.where(hit, self._noise(ctx, charset_size), cells.char)


@dataclass
class Drift(Effect):
    """A slow brightness wave crossing the art, to keep it alive between glints."""

    cycles: float = 1.0
    amplitude: float = 0.10
    wavelength: float = 0.6
    angle: float = 65.0

    def apply(self, cells: Cells, ctx: Context) -> None:
        rows, cols = cells.level.shape
        rad = math.radians(self.angle)
        x = (np.arange(cols, dtype=np.float32) + 0.5) / cols
        y = (np.arange(rows, dtype=np.float32) + 0.5) / rows
        d = x[None, :] * math.cos(rad) + y[:, None] * math.sin(rad)
        k = max(1, int(round(self.cycles)))
        wave = np.sin(2.0 * math.pi * (k * ctx.phase - d / max(self.wavelength, 1e-3)))
        cells.level = np.clip(cells.level + self.amplitude * wave, 0.0, 1.0).astype(np.float32)


@dataclass
class Twinkle(Effect):
    """Sparkle the isolated bright points of a dark field, such as a starfield.

    Picks its cells from the original quantised field, not from the running frame:
    dim, locally isolated cells, which is what a star becomes once a 12x24 px patch
    of sky is averaged into one cell. The ship or subject, being a large bright
    mass, fails the isolation test and is left alone.
    """

    density: float = 0.55
    amplitude: float = 0.45
    sharpness: float = 8.0
    min_cycles: float = 2.0
    max_cycles: float = 7.0
    floor: float = 0.02
    ceiling: float = 0.45
    isolation: float = 0.015
    tint: float = 0.6

    def __post_init__(self) -> None:
        self._plan: Dict[int, tuple] = {}

    def _stars(self, ctx: Context, level: np.ndarray) -> tuple:
        key = id(ctx.meta)
        if key not in self._plan:
            rows, cols = level.shape
            pad = np.pad(level, 1, mode="edge")
            local = sum(
                pad[y : y + rows, x : x + cols] for y in range(3) for x in range(3)
            ) / 9.0
            candidate = (
                (level >= self.floor)
                & (level <= self.ceiling)
                & ((level - local) >= self.isolation)
            )
            rng = np.random.default_rng(99173)
            active = candidate & (rng.random(level.shape) < self.density)
            lo, hi = int(self.min_cycles), max(int(self.min_cycles), int(self.max_cycles))
            cycles = rng.integers(lo, hi + 1, size=level.shape).astype(np.float32)
            offsets = rng.random(level.shape).astype(np.float32)
            self._plan[key] = (active, cycles, offsets)
        return self._plan[key]

    def apply(self, cells: Cells, ctx: Context) -> None:
        source = ctx.field.level if ctx.field is not None else cells.level
        active, cycles, offsets = self._stars(ctx, source)
        if not active.any():
            return
        wave = np.sin(2.0 * math.pi * (cycles * ctx.phase + offsets))
        pulse = np.maximum(wave, 0.0) ** self.sharpness
        boost = (self.amplitude * pulse * active).astype(np.float32)
        _brighten(cells, boost, self.tint)


EFFECTS = {"glint": Glint, "scan": Scan, "drift": Drift, "twinkle": Twinkle}


def parse_effect(spec: str) -> Effect:
    """Build an effect from a spec like `glint:period=5,strength=0.4`."""
    name, _, args = spec.partition(":")
    name = name.strip()
    if name not in EFFECTS:
        raise ValueError(f"unknown effect {name!r}; choose from {', '.join(sorted(EFFECTS))}")
    kwargs = {}
    for part in filter(None, (a.strip() for a in args.split(","))):
        key, _, value = part.partition("=")
        kwargs[key.strip()] = float(value)
    return EFFECTS[name](**kwargs)


def parse_effects(specs: List[str]) -> List[Effect]:
    """Build a list of effects from CLI specs, applied in order."""
    return [parse_effect(s) for s in specs]
