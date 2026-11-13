"""Measure a picture before animating it, and measure the motion afterwards.

Every parameter of an effect is a fact about the picture: where a thing is, which
way its surface slopes, how far it may travel between frames. Guessing them costs
renders. This reads them instead.

It takes the same flags as `asciibg` itself, plus `--find`, so the framing measured
here is exactly the framing that will be rendered. Copy the command, swap
`--find blobs` for the `--effect` it prints, and the numbers carry over untouched.

    python -m tools.probe art.png --cols 240 --rows 60 --offset-y -1 \\
        --find blobs --hue 25

    python -m tools.probe art.png --cols 240 --rows 60 --offset-y -1 \\
        --find ridge --hue 320 --band 36-46

    python -m tools.probe art.png --cols 240 --rows 60 --offset-y -1 \\
        --find motion --effect "roll:left=0.3,right=0.35,top=0.58,bottom=0.65"
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np

from asciibg.cli import build_parser
from asciibg.effects import parse_effects
from asciibg.piece import Context, Field, Meta, cells_from_field
from asciibg.quantize import field_from_image
from asciibg.ramps import DEFAULT_RAMP, ramp

# How far a moving thing may travel between frames, as a fraction of its own width.
# Overlap between one frame and the next is what makes motion read as motion, and it
# is what is lost: a barrel eleven cells wide that jumps forty lands clear of where
# it was every time, so it strobes between a few positions instead of travelling. A
# wide soft band may cross far faster and still overlap almost entirely, which is why
# this is measured against the thing's own size rather than in cells.
#
# The loop check cannot see any of this. It asks whether the loop closes, not whether
# the frames in between join up, and it calls a strobe seamless at a ratio of 1.00.
SMOOTH_TRAVEL = 0.25


def hue_mask(field: Field, hue: float, spread: float, min_level: float) -> np.ndarray:
    """Cells whose colour is within `spread` degrees of `hue` and bright enough.

    Hue rather than a hand-written RGB predicate, because one number and a tolerance
    say what "the magenta girders" or "the tan barrels" means, and a grey cell is
    excluded by its saturation rather than by luck.
    """
    r, g, b = field.rgb[..., 0], field.rgb[..., 1], field.rgb[..., 2]
    top, bottom = field.rgb.max(axis=-1), field.rgb.min(axis=-1)
    chroma = top - bottom
    safe = np.maximum(chroma, 1e-6)
    h = np.where(top == r, (g - b) / safe,
                 np.where(top == g, 2.0 + (b - r) / safe, 4.0 + (r - g) / safe)) * 60.0
    h = np.where(chroma < 1e-6, 0.0, h) % 360.0
    saturation = np.where(top < 1e-6, 0.0, chroma / np.maximum(top, 1e-6))
    off = np.abs((h - hue + 180.0) % 360.0 - 180.0)
    return (off <= spread) & (saturation > 0.25) & (field.level > min_level)


def blobs(mask: np.ndarray, gap: int, min_cells: int) -> List[Tuple[int, int, int, int, int]]:
    """Connected runs of masked cells, as (size, row0, row1, col0, col1) boxes.

    `gap` is how many blank columns a component may bridge, so 0 is ordinary
    adjacency. A quantised sprite breaks into pieces where a cell happens to fall
    between two of its bright parts, and without a bridge one barrel comes back as
    three; bridge too far and it joins the hand holding it.
    """
    rows, cols = mask.shape
    seen = np.zeros_like(mask)
    found = []
    for y in range(rows):
        for x in range(cols):
            if not mask[y, x] or seen[y, x]:
                continue
            stack, cells = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in range(-(gap + 1), gap + 2):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            if len(cells) >= min_cells:
                found.append((len(cells), min(ys), max(ys), min(xs), max(xs)))
    return sorted(found, reverse=True)


def ridge(mask: np.ndarray, band: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """The top of a masked surface at each end of its span, as (col0, row0, col1, row1).

    A girder is a surface a thing rests on, so what matters is its topmost masked row
    per column, not a line fitted through the whole beam: a fit is pulled down by the
    beam's own depth and by whatever rivets and ladders share its rows.
    """
    lo, hi = band
    tops = {}
    for col in range(mask.shape[1]):
        hit = np.where(mask[lo:hi + 1, col])[0]
        if len(hit):
            tops[col] = lo + int(hit[0])
    if len(tops) < 10:
        return None
    first, last = min(tops), max(tops)
    return first, tops[first], last, tops[last]


def motion(field: Field, meta: Meta, effects) -> List[Tuple[int, float, float]]:
    """Per effect: distinct frames, cells travelled per frame, and how wide it is.

    Distinct frames is the measure the loop check lacks, and the one that exposes a
    strobe outright: motion that outruns the frame rate lands on the same handful of
    positions every cycle, so a 360-frame loop can hold three layouts and repeat
    them.
    """
    plain = cells_from_field(field)
    out = []
    for effect in effects:
        seen, centres, widths = set(), [], []
        for i in range(meta.frame_count):
            cells = cells_from_field(field)
            effect.apply(cells, Context(meta=meta, frame=i, field=field))
            seen.add(cells.level.tobytes() + cells.char.tobytes())
            # Where it arrived, not everywhere it touched: an effect that carries
            # something also blanks where it came from, and a centroid over both
            # sits between them and halves the travel.
            moved = np.argwhere(cells.level > plain.level + 1e-6)
            if not len(moved):
                centres.append(np.nan)
                continue
            centres.append(moved[:, 1].mean())
            widths.append(moved[:, 1].max() - moved[:, 1].min() + 1)
        steps = np.abs(np.diff([c for c in centres if c == c]))
        steps = steps[steps < meta.cols / 4.0]  # drop the wrap back to the start
        out.append((len(seen),
                    float(np.median(steps)) if len(steps) else 0.0,
                    float(np.median(widths)) if widths else 0.0))
    return out


def build_probe_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.prog = "tools.probe"
    g = p.add_argument_group("probe")
    g.add_argument("--find", required=True, choices=("blobs", "ridge", "motion"),
                   help="blobs: boxes round things of one colour. ridge: the top of a "
                        "surface, as track endpoints. motion: whether the effects "
                        "given with --effect actually travel")
    g.add_argument("--hue", type=float, default=0.0, metavar="DEG",
                   help="target hue: ~25 tan, ~320 magenta, ~190 cyan")
    g.add_argument("--spread", type=float, default=25.0, metavar="DEG")
    g.add_argument("--min-level", type=float, default=0.25)
    g.add_argument("--gap", type=int, default=1,
                   help="blank columns a blob may bridge; 0 is plain adjacency")
    g.add_argument("--min-cells", type=int, default=8)
    g.add_argument("--band", default=None, metavar="A-B",
                   help="rows to search for --find ridge, e.g. 36-46")
    return p


def run(args: argparse.Namespace) -> int:
    if not args.image:
        raise SystemExit("probe reads a picture: give it an image.")
    field = field_from_image(
        args.image, args.cols, args.rows,
        aspect=args.width / args.height, zoom=args.zoom,
        offset=(args.offset_x, args.offset_y), clip=(args.clip_low, args.clip_high),
        gamma=args.gamma, saturation=args.saturation, gain=args.gain,
        value_boost=args.value_boost, sky_fill=args.sky_fill,
        center_dim=args.center_dim,
    )
    cols, rows = args.cols, args.rows

    if args.find == "motion":
        if not args.effects:
            raise SystemExit("--find motion needs at least one --effect to measure.")
        meta = Meta(name="probe", cols=cols, rows=rows, fps=args.fps, loop=args.loop,
                    charset=ramp(args.ramp or DEFAULT_RAMP))
        measured = motion(field, meta, parse_effects(args.effects))
        for spec, (distinct, step, width) in zip(args.effects, measured):
            # Only travel is flagged. Few distinct frames is not a fault on its own:
            # glint sweeps for part of its period and leaves the picture alone for
            # the rest, so most of its frames are identical and meant to be.
            share = step / width if width else 0.0
            note = (f"  <- strobes: moves {share:.0%} of its own width a frame"
                    if share > SMOOTH_TRAVEL else "")
            print(f"{distinct:4d}/{meta.frame_count} frames, {step:6.2f} cells per frame, "
                  f"{width:5.0f} wide ({share:4.0%}) {spec}{note}")
        return 0

    mask = hue_mask(field, args.hue, args.spread, args.min_level)
    if not mask.any():
        print(f"nothing within {args.spread:g} degrees of hue {args.hue:g}.", file=sys.stderr)
        return 1

    if args.find == "ridge":
        if not args.band:
            raise SystemExit("--find ridge needs --band A-B, the rows to search.")
        lo, hi = (int(v) for v in args.band.split("-"))
        found = ridge(mask, (lo, hi))
        if found is None:
            print(f"no surface in rows {lo}-{hi} at that hue.", file=sys.stderr)
            return 1
        c0, r0, c1, r1 = found
        # Things roll downhill, so the lower end is where the track has to finish.
        # A level surface has no downhill and is left in reading order.
        fall = "level" if r0 == r1 else f"downhill to the {'right' if r1 > r0 else 'left'}"
        a, b = ((c1, r1), (c0, r0)) if r0 > r1 else ((c0, r0), (c1, r1))
        print(f"rows {lo}-{hi}: top at col {c0} is row {r0}, at col {c1} is row {r1} ({fall})")
        print(f"  x0={a[0]/cols:.4f},y0={a[1]/rows:.4f},x1={b[0]/cols:.4f},y1={b[1]/rows:.4f}")
        return 0

    print(f"{'cells':>6} {'rows':>9} {'cols':>11}   box as fractions of the grid")
    for n, y0, y1, x0, x1 in blobs(mask, args.gap, args.min_cells):
        print(f"{n:6d} {y0:4d}-{y1:<4d} {x0:5d}-{x1:<5d}   "
              f"left={x0/cols:.4f},right={(x1+1)/cols:.4f},"
              f"top={y0/rows:.4f},bottom={(y1+1)/rows:.4f}")
    return 0


def main(argv=None) -> int:
    try:
        return run(build_probe_parser().parse_args(argv))
    except (RuntimeError, ValueError) as problem:
        print(f"probe: {problem}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
