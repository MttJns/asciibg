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
from .quantize import isolated_points


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
            charset_size = ctx.meta.ramp_len
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
            candidate = isolated_points(level, self.floor, self.ceiling, self.isolation)
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


@dataclass
class Sparkle(Effect):
    """Wink the bright specks of a dark surface on and off, as sun on water does.

    Twinkle lifts dim points that are already there; a highlight on water is the
    opposite case, a cell already at the top of the ramp, where adding brightness
    does nothing. This one scales instead, so a speck drops to `dim` and comes back,
    which reads as a catch of light rather than a glow. `top` and `bottom` keep it
    to a band of the frame, in fractions of the height, so the sky and the subject
    below stay still.
    """

    density: float = 0.7
    dim: float = 0.15
    gain: float = 0.0
    sharpness: float = 1.6
    min_cycles: float = 3.0
    max_cycles: float = 9.0
    floor: float = 0.25
    ceiling: float = 1.0
    isolation: float = 0.02
    tint: float = 0.35
    top: float = 0.0
    bottom: float = 1.0
    left: float = 0.0
    right: float = 1.0

    def __post_init__(self) -> None:
        self._plan: Dict[int, tuple] = {}

    def _band(self, rows: int, cols: int) -> np.ndarray:
        y = (np.arange(rows, dtype=np.float32) + 0.5) / rows
        x = (np.arange(cols, dtype=np.float32) + 0.5) / cols
        return ((y >= self.top) & (y < self.bottom))[:, None] & (
            (x >= self.left) & (x < self.right)
        )[None, :]

    def _specks(self, ctx: Context, level: np.ndarray) -> tuple:
        key = id(ctx.meta)
        if key not in self._plan:
            rows, cols = level.shape
            candidate = isolated_points(level, self.floor, self.ceiling, self.isolation)
            candidate &= self._band(rows, cols)
            rng = np.random.default_rng(5150021)
            active = candidate & (rng.random(level.shape) < self.density)
            lo, hi = int(self.min_cycles), max(int(self.min_cycles), int(self.max_cycles))
            cycles = rng.integers(lo, hi + 1, size=level.shape).astype(np.float32)
            offsets = rng.random(level.shape).astype(np.float32)
            self._plan[key] = (active, cycles, offsets)
        return self._plan[key]

    def apply(self, cells: Cells, ctx: Context) -> None:
        source = ctx.field.level if ctx.field is not None else cells.level
        active, cycles, offsets = self._specks(ctx, source)
        if not active.any():
            return
        wave = 0.5 + 0.5 * np.sin(2.0 * math.pi * (cycles * ctx.phase + offsets))
        pulse = wave ** self.sharpness
        scale = np.where(active, self.dim + (1.0 - self.dim) * pulse, 1.0).astype(np.float32)
        cells.level = np.clip(cells.level * scale, 0.0, 1.0)
        if self.gain > 0.0:
            _brighten(cells, (self.gain * pulse * active).astype(np.float32), self.tint)
        elif self.tint > 0.0:
            t = np.clip(self.tint * pulse * active, 0.0, 1.0)[..., None].astype(np.float32)
            cells.rgb = np.clip(cells.rgb + (1.0 - cells.rgb) * t, 0.0, 1.0)


@dataclass
class Flag(Effect):
    """Ripple a flag on its pole, as a breeze does.

    The cells inside the box are displaced vertically, by a wave running out along
    the cloth. Travel grows with distance from `pole`, a fraction across the box, so
    the hoist stays pinned and the fly end moves most. Each column samples the
    original field, which is what lets a cell that moves away show the sky behind it
    rather than a smear of itself.

    `amplitude` is in cells and `waves` is wavelengths along the cloth. One cell of
    travel is already a lot at this scale: a flag in the frame is a few cells wide,
    and anything more reads as a flag being shaken rather than a breeze.
    """

    left: float = 0.0
    right: float = 1.0
    top: float = 0.0
    bottom: float = 1.0
    amplitude: float = 1.0
    waves: float = 1.0
    cycles: float = 4.0
    pole: float = 0.0

    def apply(self, cells: Cells, ctx: Context) -> None:
        rows, cols = cells.level.shape
        x0, x1 = int(round(self.left * cols)), int(round(self.right * cols))
        y0, y1 = int(round(self.top * rows)), int(round(self.bottom * rows))
        x0, x1 = max(0, x0), min(cols, x1)
        y0, y1 = max(0, y0), min(rows, y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return

        source = ctx.field if ctx.field is not None else None
        if source is None:
            return

        span = float(x1 - x0)
        t = (np.arange(x0, x1, dtype=np.float32) + 0.5 - x0) / span
        grip = np.abs(t - self.pole) / max(self.pole, 1.0 - self.pole, 1e-6)
        k = max(1.0, round(self.cycles))
        wave = np.sin(2.0 * math.pi * (k * ctx.phase - self.waves * t))
        shifts = np.rint(self.amplitude * grip * wave).astype(int)

        y = np.arange(y0, y1)
        for i, x in enumerate(range(x0, x1)):
            shift = int(shifts[i])
            if shift == 0:
                continue
            src = np.clip(y - shift, 0, rows - 1)
            cells.level[y0:y1, x] = source.level[src, x]
            cells.rgb[y0:y1, x] = source.rgb[src, x]


# A barrel at the scale the picture draws one: the stacked barrels beside Donkey Kong
# measure about ten cells across and six deep on a 240x60 grid, so a sprite any
# smaller reads as a bead rather than as the same object rolling.
#
# The staves are what turn. Three of them sit every third column of a nine-wide body,
# and each pose shifts them one column on, so the set wraps back to where it began
# after three poses and the barrel keeps the same number of staves throughout. A
# cycle that gained or lost one would read as flicker rather than as rotation.
BARREL_BODY = ("|==", "=|=", "==|")
BARREL_POSES = tuple(
    (
        " ========= ",
        f"({body * 3})",
        f"({body * 3})",
        f"({body * 3})",
        " ========= ",
    )
    for body in BARREL_BODY
)


@dataclass
class Roll(Effect):
    """Barrels rolling along a girder, as Donkey Kong throws them.

    The track is a straight line from (`x0`, `y0`) to (`x1`, `y1`), in fractions of
    the grid, and `count` barrels ride it evenly spaced. Each one's position is the
    loop phase plus its own share of the track, taken modulo one, so at the end of
    the loop every barrel stands exactly where it started and the wrap costs nothing.

    Drawn as characters rather than brightness, so `charset_extra` asks for glyphs
    the ramp has no room for. `lift` is the gap in cells between the track and the
    bottom of the sprite, since a barrel sits on top of a girder rather than in it.
    """

    charset_extra = "()|="

    x0: float = 0.0
    y0: float = 0.5
    x1: float = 1.0
    y1: float = 0.5
    count: float = 3.0
    cycles: float = 1.0
    spins: float = 10.0
    lift: float = 1.0
    r: float = 1.0
    g: float = 0.62
    b: float = 0.24

    def apply(self, cells: Cells, ctx: Context) -> None:
        rows, cols = cells.level.shape
        n = max(1, int(round(self.count)))
        k = max(1.0, round(self.cycles))
        colour = np.array([self.r, self.g, self.b], dtype=np.float32)

        for i in range(n):
            t = (k * ctx.phase + i / n) % 1.0
            x = (self.x0 + (self.x1 - self.x0) * t) * cols
            y = (self.y0 + (self.y1 - self.y0) * t) * rows - self.lift
            pose = BARREL_POSES[int(t * max(1.0, round(self.spins)) * len(BARREL_POSES))
                                % len(BARREL_POSES)]
            # The track is the ground the barrel rests on, so the sprite hangs above
            # it: its last line lands on the lifted row and the rest stack upwards.
            base = int(round(y)) - (len(pose) - 1)
            start = int(round(x)) - len(pose[0]) // 2
            for dy, line in enumerate(pose):
                row = base + dy
                if not 0 <= row < rows:
                    continue
                for dx, ch in enumerate(line):
                    col = start + dx
                    if ch == " " or not 0 <= col < cols:
                        continue
                    index = ctx.meta.charset.find(ch)
                    if index < 0:
                        continue
                    cells.char[row, col] = index
                    cells.level[row, col] = 1.0
                    cells.rgb[row, col] = colour


EFFECTS = {"glint": Glint, "scan": Scan, "drift": Drift, "twinkle": Twinkle,
           "sparkle": Sparkle, "flag": Flag, "roll": Roll}


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


def charset_extra(effects: List[Effect]) -> str:
    """Characters the effects draw with that a brightness ramp would not carry."""
    seen = ""
    for effect in effects:
        for ch in getattr(effect, "charset_extra", ""):
            if ch not in seen:
                seen += ch
    return seen
