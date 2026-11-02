# asciibg

Turn a base image into a seamlessly looping, pixelated ASCII animation, exported as
an MP4 for Zoom virtual backgrounds and a GIF for the web. The look follows
[ascii.rest](https://ascii.rest): the art is quantised once into a fixed grid of
characters, and motion is a layer over it.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
```

Needs `ffmpeg` on PATH. Uses a system monospace font (Menlo on macOS); no download.

## Use

```bash
.venv/bin/python -m asciibg.cli photo.jpg -o out
```

Writes `out/photo.mp4` (1920x1080, H.264, yuv420p), `out/photo.gif`,
`out/photo.png` (first frame, as a still background) and `out/photo-sheet.png`
(a 3x3 contact sheet with the speaker's safe zone marked).

```bash
.venv/bin/python -m asciibg.cli photo.jpg \
  --loop 12 --ramp dot --center-dim 0.5 \
  --effect "glint:period=5,strength=0.6" \
  --effect "scan:period=9" \
  --effect "drift:amplitude=0.08"
```

## How it works

1. **Quantise once.** Cover-crop to 16:9, box-downsample to `cols x rows`, take each
   cell's mean colour and luminance. Luminance is normalised to the image's own 2nd
   and 98th percentiles, so the ramp spans the picture's real range.
2. **Pick characters.** The ramp is sorted by *measured ink coverage in the chosen
   font*, not by its written order. Conventional ramps are not monotonic: in Menlo
   `-` carries less ink than `:`, and `%` less than `#`. `--no-sort-ramp` keeps the
   written order.
3. **Layer motion.** `glint` sweeps a bright band across, `scan` runs a line down
   that scrambles the characters it passes, `drift` is a slow brightness wave.
   The grid itself never changes.
4. **Render.** Every character is rasterised once at the exact cell size into an
   alpha mask, then blitted tinted by each cell's colour. Nothing is ever resampled,
   which is what keeps the glyph edges crisp.
5. **Encode.** Raw frames pipe straight into ffmpeg; the GIF is made from the video.

## Geometry

The grid must divide the frame exactly, so cells land on whole pixels. The default
160x45 into 1920x1080 gives 12x24 px cells, and Menlo at size 20 has an advance of
exactly 12 px and a height of exactly 24. Other exact fits: 192x54, 240x60, 128x36.

## Looping

Every effect is a function of the loop phase alone, built from periodic terms or
from tables indexed by frame number. Frame count is exactly `loop * fps`, and frame
`N` is never rendered because it *is* frame 0. Each render reports a loop check that
compares the last-to-first step against the median frame-to-frame step; the CLI
exits non-zero if the wrap breaks.

## Making it work as a Zoom background

- Zoom takes MP4/MOV video and JPG/PNG stills, at up to 1920x1080.
- Your head and shoulders cover roughly the centre of the frame. `--center-dim 0.5`
  fades that area so your face reads against it; the contact sheet marks the zone.
- Zoom's "mirror my video" flips your self-view, so lettering reads backwards to you
  but correctly to everyone else. Prefer art without text.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```
