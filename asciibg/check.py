"""Prove the loop is seamless, and make a contact sheet to look at."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageDraw


def _thumb(frame: np.ndarray, step: int = 8) -> np.ndarray:
    return frame[::step, ::step].astype(np.float32).mean(axis=2)


@dataclass
class LoopCheck:
    """Accumulates inter-frame differences, including the wrap from last back to first."""

    deltas: List[float] = field(default_factory=list)
    _first: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _prev: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def add(self, frame: np.ndarray) -> None:
        small = _thumb(frame)
        if self._first is None:
            self._first = small
        else:
            self.deltas.append(float(np.abs(small - self._prev).mean()))
        self._prev = small

    @property
    def wrap_delta(self) -> float:
        return float(np.abs(self._first - self._prev).mean())

    @property
    def median_delta(self) -> float:
        return float(np.median(self.deltas)) if self.deltas else 0.0

    def report(self, tolerance: float = 3.0) -> str:
        """A one-line verdict. The wrap should look like any other frame step."""
        med, wrap = self.median_delta, self.wrap_delta
        if med < 1e-6:
            verdict = "seamless (static)" if wrap < 1e-6 else "BREAK"
            return f"loop: {verdict} wrap={wrap:.4f} median={med:.4f}"
        ratio = wrap / med
        verdict = "seamless" if ratio <= tolerance else "BREAK"
        return f"loop: {verdict} wrap={wrap:.4f} median={med:.4f} ratio={ratio:.2f}x"

    def ok(self, tolerance: float = 3.0) -> bool:
        med = self.median_delta
        return self.wrap_delta < 1e-6 if med < 1e-6 else self.wrap_delta / med <= tolerance


def contact_sheet(frames: Sequence[np.ndarray], path: Path, *, safe_zone: bool = True) -> None:
    """Write a 3x3 sheet of sampled frames, with the speaker's safe zone marked."""
    if not frames:
        return
    cols = 3
    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    tw, th = w // 4, h // 4
    sheet = Image.new("RGB", (tw * cols, th * rows), (18, 18, 18))
    for i, frame in enumerate(frames):
        tile = Image.fromarray(frame).resize((tw, th), Image.LANCZOS)
        if safe_zone:
            draw = ImageDraw.Draw(tile, "RGBA")
            draw.ellipse(
                [tw * 0.20, th * 0.03, tw * 0.80, th * 1.13],
                outline=(255, 80, 80, 160), width=2,
            )
        sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
    sheet.save(path)
