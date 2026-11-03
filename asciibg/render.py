"""Rasterise a cell grid to RGB frames with a pre-built glyph atlas.

Every character is rasterised once, at the exact cell size, into an alpha mask.
A frame is then a blit of those masks tinted by each cell's colour, which keeps
glyph edges pixel-crisp because nothing is ever resampled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Monospace faces worth using, best first, by file name rather than full path: the
# same font sits in different directories on each platform, and distributions move
# them between releases. Every one of these is fixed-pitch, which the grid requires.
FONT_FILES = (
    ("Menlo.ttc", 0),                   # macOS
    ("SFNSMono.ttf", 0),
    ("Andale Mono.ttf", 0),
    ("DejaVuSansMono.ttf", 0),          # most Linux distributions
    ("LiberationMono-Regular.ttf", 0),
    ("NotoSansMono-Regular.ttf", 0),
    ("UbuntuMono-R.ttf", 0),
    ("consola.ttf", 0),                 # Windows
    ("lucon.ttf", 0),
    ("cour.ttf", 0),
)


def font_dirs() -> Tuple[str, ...]:
    """The directories this platform keeps fonts in, that actually exist."""
    candidates = (
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
    )
    return tuple(d for d in candidates if os.path.isdir(d))


def find_font() -> Tuple[str, int]:
    """Search this platform's font directories for the best monospace face.

    Returns (path, face index). Preference is by position in `FONT_FILES`, not by
    which directory is searched first, so a machine with both Menlo and DejaVu gets
    Menlo wherever each one happens to live.
    """
    rank = {name.lower(): i for i, (name, _) in enumerate(FONT_FILES)}
    best = None
    for directory in font_dirs():
        for root, _, files in os.walk(directory):
            for name in files:
                i = rank.get(name.lower())
                if i is not None and (best is None or i < best[0]):
                    best = (i, os.path.join(root, name))
    if best is None:
        looked = ", ".join(font_dirs()) or "no font directory on this system"
        raise RuntimeError(
            "no monospace font found. Pass one with --font /path/to/Mono.ttf.\n"
            f"Looked for {', '.join(n for n, _ in FONT_FILES)} in: {looked}"
        )
    return best[1], FONT_FILES[best[0]][1]


def check_monospace(face: "ImageFont.FreeTypeFont", path: str) -> None:
    """Refuse a proportional font: the whole grid assumes one advance width."""
    widths = {round(face.getlength(ch), 3) for ch in "iWM.@"}
    if len(widths) > 1:
        raise RuntimeError(
            f"{path} is not monospace: its characters have advances {sorted(widths)}. "
            "Every cell is one advance wide, so a proportional font shears the grid. "
            "Pass a fixed-pitch font with --font."
        )


def fit_font_size(path: str, index: int, cell_w: int, max_size: int = 200) -> int:
    """Largest size whose advance width is at most `cell_w`, so glyphs never overlap."""
    best = 1
    for size in range(1, max_size):
        font = ImageFont.truetype(path, size, index=index)
        if font.getlength("@") > cell_w:
            break
        best = size
    return best


@dataclass
class Atlas:
    """Alpha masks for a charset, one per character, at exactly cell_w x cell_h."""

    charset: str
    masks: np.ndarray  # uint8 (n, cell_h, cell_w)
    cell_w: int
    cell_h: int

    @property
    def size(self) -> int:
        return len(self.charset)


def build_atlas(
    charset: str,
    cell_w: int,
    cell_h: int,
    *,
    font: Optional[str] = None,
    font_index: int = 0,
    font_size: Optional[int] = None,
) -> Atlas:
    """Rasterise `charset` into per-character alpha masks sized to the cell."""
    path, index = (font, font_index) if font else find_font()
    size = font_size or fit_font_size(path, index, cell_w)
    face = ImageFont.truetype(path, size, index=index)
    check_monospace(face, path)
    ascent, descent = face.getmetrics()
    advance = face.getlength("@")

    # Centre the glyph box in the cell: terminals use a line height above the font size.
    x0 = (cell_w - advance) / 2.0
    baseline = (cell_h - (ascent + descent)) / 2.0 + ascent

    masks = np.zeros((len(charset), cell_h, cell_w), dtype=np.uint8)
    for i, ch in enumerate(charset):
        if ch == " ":
            continue
        tile = Image.new("L", (cell_w, cell_h), 0)
        ImageDraw.Draw(tile).text((x0, baseline), ch, fill=255, font=face, anchor="ls")
        masks[i] = np.asarray(tile, dtype=np.uint8)
    return Atlas(charset=charset, masks=masks, cell_w=cell_w, cell_h=cell_h)


def render_frame(
    char_idx: np.ndarray,
    rgb: np.ndarray,
    atlas: Atlas,
    background: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    """Composite one frame. Returns uint8 (rows*cell_h, cols*cell_w, 3).

    Works a cell-row at a time: the whole frame as one array would be hundreds of
    megabytes, a single row is a few hundred kilobytes.
    """
    rows, cols = char_idx.shape
    ch, cw = atlas.cell_h, atlas.cell_w
    bg = np.asarray(background, dtype=np.uint16)
    fg = np.clip(rgb * 255.0, 0, 255).astype(np.uint16)

    out = np.empty((rows * ch, cols * cw, 3), dtype=np.uint8)
    for y in range(rows):
        alpha = atlas.masks[char_idx[y]].astype(np.uint16)  # (cols, ch, cw)
        a = alpha[:, :, :, None]
        colour = fg[y][:, None, None, :]  # (cols, 1, 1, 3)
        band = (colour * a + bg * (255 - a)) // 255  # (cols, ch, cw, 3)
        out[y * ch : (y + 1) * ch] = band.transpose(1, 0, 2, 3).reshape(ch, cols * cw, 3)
    return out


def sort_by_ink(atlas: Atlas, limit: int = 0) -> Atlas:
    """Reorder a charset by measured ink coverage, lightest first.

    Conventional ramps are not monotonic in every font: in Menlo, `-` carries less
    ink than `:` and `%` less than `#`. Sorting makes brightness actually increase
    along the ramp. `limit` sorts only that many leading characters, leaving glyphs
    a scene added after the ramp where they are.
    """
    n = limit or atlas.size
    ink = atlas.masks.reshape(atlas.size, -1).mean(axis=1)
    order = np.concatenate([np.argsort(ink[:n], kind="stable"), np.arange(n, atlas.size)])
    return Atlas(
        charset="".join(atlas.charset[i] for i in order),
        masks=atlas.masks[order],
        cell_w=atlas.cell_w,
        cell_h=atlas.cell_h,
    )
