"""The animation contract: a fixed cell field, plus per-frame cell state.

Mirrors ascii.rest's piece shape (setup once, draw per frame) so the same art can
later be exported as a .ts piece. Setup produces a `Field`; each frame starts from
a copy of it as `Cells`, which effects mutate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Meta:
    """Facts about a piece: its grid, its timing and the ramp it draws with."""

    name: str
    cols: int
    rows: int
    fps: int
    loop: float
    charset: str
    note: str = ""

    @property
    def frame_count(self) -> int:
        """Number of distinct frames in one loop. Frame `frame_count` is frame 0."""
        return int(round(self.loop * self.fps))


@dataclass
class Field:
    """The fixed art: one brightness and one colour per cell, quantised once."""

    level: np.ndarray  # float32 (rows, cols) in 0..1
    rgb: np.ndarray  # float32 (rows, cols, 3) in 0..1

    @property
    def shape(self) -> Tuple[int, int]:
        return self.level.shape  # type: ignore[return-value]


@dataclass
class Cells:
    """One frame's mutable cell state, before characters are chosen."""

    level: np.ndarray  # float32 (rows, cols) in 0..1
    rgb: np.ndarray  # float32 (rows, cols, 3) in 0..1
    char: np.ndarray  # int16 (rows, cols); >= 0 overrides the ramp, -1 means quantise


@dataclass(frozen=True)
class Context:
    """What an effect needs to know about the piece and the current frame."""

    meta: Meta
    frame: int

    @property
    def t(self) -> float:
        """Time in seconds within the loop."""
        return self.frame / self.meta.fps

    @property
    def phase(self) -> float:
        """Position in the loop, in 0..1. Exactly 0 at the first frame."""
        return self.frame / self.meta.frame_count


def cells_from_field(field: Field) -> Cells:
    """Start a frame from the fixed field. Copies, so effects may mutate freely."""
    return Cells(
        level=field.level.copy(),
        rgb=field.rgb.copy(),
        char=np.full(field.level.shape, -1, dtype=np.int16),
    )


def resolve_chars(cells: Cells, charset_size: int) -> np.ndarray:
    """Pick a character index per cell: an override if set, else the ramp position."""
    quantised = np.clip(
        np.rint(np.clip(cells.level, 0.0, 1.0) * (charset_size - 1)), 0, charset_size - 1
    ).astype(np.int16)
    return np.where(cells.char >= 0, np.clip(cells.char, 0, charset_size - 1), quantised)
