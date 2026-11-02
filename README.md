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
   that scrambles the characters it passes, `drift` is a slow brightness wave, and
   `twinkle` sparkles isolated dim points such as a starfield. The grid itself
   never changes.
4. **Render.** Every character is rasterised once at the exact cell size into an
   alpha mask, then blitted tinted by each cell's colour. Nothing is ever resampled,
   which is what keeps the glyph edges crisp.
5. **Encode.** Raw frames pipe straight into ffmpeg; the GIF is made from the video.

## Subjects on a dark background

A picture that is mostly background, such as a ship on a night sky, puts its subject
at the bottom of the ramp, because the sky dominates the histogram. Two controls fix
it:

- `--clip-low 70` maps everything below the 70th percentile to empty, so the sky goes
  black and the subject spreads across the whole ramp.
- `--value-boost 0.85` lifts each cell's colour towards full value, keeping its hue.
  Without it a dim cell gets both a sparse character and a dark colour, and reads as
  doubly dark.

`twinkle` picks its cells from the quantised field: dim, locally isolated ones, which
is what a star becomes once a 12x24 px patch of sky averages into a single cell. A
large bright subject fails the isolation test, so it never sparkles.

```bash
.venv/bin/python -m asciibg.cli xwing.jpg --loop 10 \
  --cols 240 --rows 60 --zoom 0.45 --offset-x -0.9 --offset-y 0.45 \
  --clip-low 70 --clip-high 99.5 --gamma 0.7 --value-boost 0.85 --saturation 1.4 \
  --effect "twinkle:amplitude=0.45,density=0.6"
```

## Shrinking a subject: zoom below 1

`--zoom` below 1 makes the window larger than the image, so the subject shrinks and
the rest is padding. Two things have to be right or the result gives itself away:

- **The padding must pass for more of the same background.** Copying only the stars
  is not enough. A photographed night sky is not black but a dim haze, so stars on
  true black read as a visible rectangle. Every padded cell instead takes a whole
  random background cell, brightness and colour together, which reproduces the haze,
  the star density and the colour spread at once.
- **Percentiles must come from the real pixels only.** The padding is black, and
  counting it drags the low percentile to zero, which silently disables `--clip-low`
  exactly when zooming out has made it matter most.

Both are tested: one case asserts the clip still bites when most of the frame is
padding, another that the padded cells match the real sky's mean brightness.

## Keeping the subject out from behind your head

`--zoom` and `--offset-x/--offset-y` pan the subject away from the centre. A source
already at 16:9 has no pan room at all, so every bit of it has to be bought by
zooming in, and that crops the edges: judge the trade rather than guessing.

Measure it instead of eyeballing it. Build a head-and-shoulders mask, then count how
much of the subject falls inside it:

```python
x = (np.arange(COLS) + 0.5) / COLS
y = (np.arange(ROWS) + 0.5) / ROWS
head = (((x[None, :] - 0.5) / 0.19) ** 2 + ((y[:, None] - 0.72) / 0.52) ** 2) <= 1.0
ship = field.level > 0.5
hidden = ship[head].sum() / ship.sum()
```

For the X-wing above, centred put 50% of the ship behind the speaker's head. Sweeping
zoom and offset found `--zoom 1.5 --offset-x 0.9 --offset-y 1.0`, which drops it to
3.7% by lifting the ship into a banner across the top, at the cost of the nose
running off the left edge.

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
