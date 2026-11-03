"""Command line: a base image in, a looping ASCII background out."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .check import LoopCheck, contact_sheet
from .effects import parse_effects
from .encode import Mp4Writer, mp4_to_gif, probe, save_png
from .piece import Context, Meta, cells_from_field, resolve_chars
from .prompt import scene_from_prompt
from .quantize import _center_mask, field_from_image
from .ramps import DEFAULT_RAMP, RAMPS, ramp
from .render import build_atlas, render_frame, sort_by_ink
from .scenes import SCENES, parse_scene

SHEET_FRAMES = 9


def parse_colour(text: str) -> Tuple[int, int, int]:
    """Parse a #rrggbb background colour."""
    value = text.lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError(f"expected #rrggbb, got {text!r}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asciibg",
        description="Turn a base image into a looping ASCII background for Zoom.",
    )
    p.add_argument("image", nargs="?", help="base image (jpg, png, ...)")
    source = p.add_argument_group("source (pick one)")
    source.add_argument("--scene", metavar="SPEC",
                        help="a procedural scene instead of an image, e.g. "
                             "matrix:density=0.7,palette=amber")
    source.add_argument("--prompt", metavar="TEXT",
                        help="describe what you want and let it pick and tune a scene")
    source.add_argument("--list-scenes", action="store_true",
                        help="list the scenes and what each answers to, then exit")
    p.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    p.add_argument("-n", "--name", help="basename for the outputs (default: the image's)")

    grid = p.add_argument_group("grid")
    grid.add_argument("--width", type=int, default=1920)
    grid.add_argument("--height", type=int, default=1080)
    grid.add_argument("--cols", type=int, default=160)
    grid.add_argument("--rows", type=int, default=45)
    grid.add_argument("--zoom", type=float, default=1.0,
                      help="above 1 crops tighter; below 1 shrinks the subject and pads "
                           "around it, with the padding filled from the image's own sky")
    grid.add_argument("--no-sky-fill", dest="sky_fill", action="store_false",
                      help="leave padding black instead of extending the background into it")
    grid.add_argument("--offset-x", type=float, default=0.0, metavar="N",
                      help="-1..1, pan the crop so the subject sits off-centre")
    grid.add_argument("--offset-y", type=float, default=0.0, metavar="N")

    timing = p.add_argument_group("timing")
    timing.add_argument("--fps", type=int, default=30)
    timing.add_argument("--loop", type=float, default=12.0, help="loop length in seconds")

    look = p.add_argument_group("look")
    look.add_argument("--ramp", default=None, choices=sorted(RAMPS),
                      help=f"character ramp (default: {DEFAULT_RAMP}, or the scene's own)")
    look.add_argument("--no-sort-ramp", dest="sort_ramp", action="store_false",
                      help="keep the ramp's written order instead of sorting by ink coverage")
    look.add_argument("--clip-low", type=float, default=2.0, metavar="PCT",
                      help="percentile mapped to the empty end of the ramp; raise it "
                           "for a subject on a mostly dark background (default: 2)")
    look.add_argument("--clip-high", type=float, default=98.0, metavar="PCT")
    look.add_argument("--gamma", type=float, default=1.0)
    look.add_argument("--value-boost", type=float, default=0.0,
                      help="0..1, lift each cell's colour towards full value, keeping hue")
    look.add_argument("--saturation", type=float, default=1.0)
    look.add_argument("--gain", type=float, default=1.0)
    look.add_argument("--center-dim", type=float, default=0.0,
                      help="0..1, fade the middle where a speaker's head sits")
    look.add_argument("--bg", type=parse_colour, default=(0, 0, 0))

    motion = p.add_argument_group("motion")
    motion.add_argument("--effect", action="append", dest="effects", metavar="SPEC",
                        help="repeatable, e.g. glint:period=5,strength=0.6 (default: glint, drift)")

    font = p.add_argument_group("font")
    font.add_argument("--font", help="path to a monospace font")
    font.add_argument("--font-index", type=int, default=0)
    font.add_argument("--font-size", type=int)

    out = p.add_argument_group("output")
    out.add_argument("--crf", type=int, default=18)
    out.add_argument("--no-gif", dest="gif", action="store_false")
    out.add_argument("--gif-width", type=int, default=720)
    out.add_argument("--gif-fps", type=int, default=15)
    out.add_argument("--no-sheet", dest="sheet", action="store_false")
    out.add_argument("-q", "--quiet", action="store_true")
    return p


def _check_grid(args: argparse.Namespace) -> Tuple[int, int]:
    if args.width % args.cols or args.height % args.rows:
        cw, ch = args.width / args.cols, args.height / args.rows
        raise SystemExit(
            f"grid does not divide the frame: {args.cols}x{args.rows} cells into "
            f"{args.width}x{args.height} gives {cw:.2f}x{ch:.2f} px cells.\n"
            "Pick a grid that divides exactly, e.g. 160x45, 192x54, 240x60, or 128x36."
        )
    return args.width // args.cols, args.height // args.rows


def list_scenes() -> int:
    """Print the catalogue and the words each scene answers to."""
    for name in sorted(SCENES):
        cls = SCENES[name]
        print(f"{name:11s} {cls.__doc__.splitlines()[0]}")
        print(f"{'':11s} words: {', '.join(cls.tags)}")
    return 0


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_scenes:
        return list_scenes()

    chosen = [n for n, v in (("image", args.image), ("--scene", args.scene),
                             ("--prompt", args.prompt)) if v]
    if len(chosen) != 1:
        raise SystemExit(
            "give exactly one source: an image, --scene, or --prompt "
            f"(got {', '.join(chosen) or 'none'}).\n"
            "Run with --list-scenes to see the scenes."
        )
    cell_w, cell_h = _check_grid(args)

    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))

    scene = None
    if args.prompt:
        scene, why = scene_from_prompt(args.prompt)
        say(f"prompt -> {why}")
    elif args.scene:
        scene = parse_scene(args.scene)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_name = Path(args.image).stem if args.image else type(scene).__name__.lower()
    name = args.name or default_name
    mp4_path, gif_path = out_dir / f"{name}.mp4", out_dir / f"{name}.gif"
    png_path, sheet_path = out_dir / f"{name}.png", out_dir / f"{name}-sheet.png"

    ramp_name = args.ramp or getattr(scene, "ramp_name", None) or DEFAULT_RAMP
    base_ramp = ramp(ramp_name)
    extra = "".join(c for c in getattr(scene, "charset_extra", "") if c not in base_ramp)
    atlas = build_atlas(base_ramp + extra, cell_w, cell_h, font=args.font,
                        font_index=args.font_index, font_size=args.font_size)
    if args.sort_ramp:
        atlas = sort_by_ink(atlas, limit=len(base_ramp))

    meta = Meta(name=name, cols=args.cols, rows=args.rows, fps=args.fps,
                loop=args.loop, charset=atlas.charset, ramp_size=len(base_ramp))

    field = None
    if scene is not None:
        scene.setup(meta)
        dim = args.center_dim * _center_mask(args.cols, args.rows) if args.center_dim else None
    else:
        field = field_from_image(
            args.image, args.cols, args.rows,
            aspect=args.width / args.height, zoom=args.zoom,
            offset=(args.offset_x, args.offset_y), clip=(args.clip_low, args.clip_high),
            gamma=args.gamma, saturation=args.saturation, gain=args.gain,
            value_boost=args.value_boost, sky_fill=args.sky_fill,
            center_dim=args.center_dim,
        )
        dim = None

    effects = parse_effects(args.effects) if args.effects else (
        [] if scene is not None else parse_effects(["glint", "drift"])
    )

    total = meta.frame_count
    say(f"{name}: {args.cols}x{args.rows} cells, {cell_w}x{cell_h} px each, "
        f"{total} frames at {args.fps} fps ({args.loop:g}s loop)")
    say(f"ramp {ramp_name} {atlas.charset!r}")

    check = LoopCheck()
    sheet: List[np.ndarray] = []
    sheet_at = {round(i * total / SHEET_FRAMES) for i in range(SHEET_FRAMES)}

    started = time.time()
    with Mp4Writer(mp4_path, args.width, args.height, args.fps, crf=args.crf) as writer:
        for i in range(total):
            ctx = Context(meta=meta, frame=i, field=field)
            if scene is not None:
                cells = scene.render(ctx)
                if dim is not None:
                    cells.level = (cells.level * (1.0 - dim)).astype(np.float32)
            else:
                cells = cells_from_field(field)
            for effect in effects:
                effect.apply(cells, ctx)
            frame = render_frame(resolve_chars(cells, meta.ramp_len, atlas.size), cells.rgb, atlas, args.bg)

            writer.write(frame)
            check.add(frame)
            if i == 0:
                save_png(frame, png_path)
            if i in sheet_at:
                sheet.append(frame.copy())
            if not args.quiet and (i + 1) % 30 == 0:
                say(f"  {i + 1}/{total} frames")

    say(f"rendered {total} frames in {time.time() - started:.1f}s")
    say(check.report())
    say(f"mp4  {mp4_path}  [{probe(mp4_path)}]")

    if args.gif:
        mp4_to_gif(mp4_path, gif_path, width=args.gif_width, fps=args.gif_fps)
        say(f"gif  {gif_path}  [{gif_path.stat().st_size // 1024} kB]")
    say(f"png  {png_path}")
    if args.sheet:
        contact_sheet(sheet, sheet_path)
        say(f"sheet {sheet_path}")

    return 0 if check.ok() else 2


if __name__ == "__main__":
    raise SystemExit(main())
