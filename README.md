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

## Scenes: backgrounds without an image

Scenes are computed per frame rather than quantised from a picture, the way
ascii.rest's own scenes are. Pick one directly, or describe what you want:

```bash
.venv/bin/python -m asciibg.cli --scene matrix:density=0.7,palette=amber -o out
.venv/bin/python -m asciibg.cli --prompt "slow heavy snow on a cold blue night" -o out
.venv/bin/python -m asciibg.cli --list-scenes
```

| scene | what it draws |
|---|---|
| `starfield` | layers of stars drifting past, nearer ones faster |
| `rain` | streaks falling at a slant (also snow, ash) |
| `waves` | interfering ripples, like light on water |
| `nebula` | slow clouds of looping noise |
| `grid` | a perspective floor running away to a horizon |
| `matrix` | columns of characters falling, brightest at the head |

Palettes: `ice`, `amber`, `green`, `ember`, `violet`, `mono`.

`--prompt` selects and tunes a scene; it does not generate one. A fixed catalogue
that is known to loop and known to read as a background is worth more than free-form
generation that may do neither. Colour words set the palette, pace words
(`fast`/`slow`) and quantity words (`heavy`/`sparse`) turn the knobs, and a prompt
matching nothing falls back to `nebula` and says so.

### Two things scenes needed from the renderer

**A ramp that suits the field.** A smooth, continuous scene drawn with the shade
ramp reads as horizontal stripes, because `-`, `=` and `:` are all horizontal
strokes. Blocks and dots have no direction, so `nebula` and `waves` default to the
`block` ramp. This was the single biggest difference in how they look; `--ramp`
still overrides it.

**Glyphs beyond the ramp.** `rain` wants `|` and `matrix` wants letters, none of
which belong in a brightness ramp. The atlas carries them *after* the ramp, and
`Meta.ramp_size` records where the ramp ends: brightness quantises into the leading
characters only, sorting by ink touches only those, and a character override may
name anything in the atlas.

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
   `twinkle` sparkles isolated dim points such as a starfield, `sparkle` winks the
   bright ones, such as sun on water, and `flag` ripples a flag on its pole. The
   grid itself never changes.
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

## Bright highlights: `sparkle` rather than `twinkle`

The two look alike and work opposite ways. `twinkle` *adds* brightness to cells that
are dim, which is what a star is. A highlight on water is the other case: the cell is
already at the top of the ramp, so adding to it does nothing at all. `sparkle`
*scales* instead, dropping a speck to `dim` and bringing it back, which reads as a
catch of light rather than a glow.

`top`, `bottom`, `left` and `right` hold it to a box of the frame, in fractions of
the height and width, so the sky above and the subject standing in front of the water
stay still. A subject that sits *inside* the band, such as a car on a dock, is
excluded by column: run one sparkle down each side of it rather than one across.

```bash
.venv/bin/python -m asciibg.cli seafront.png --loop 10 \
  --clip-low 68 --clip-high 99 --value-boost 0.7 \
  --effect "sparkle:top=0.22,bottom=0.72,density=0.75,dim=0.3,isolation=0.08"
```

Raise `isolation` to keep it off a solid bright mass that happens to sit in the band,
such as a hull on the horizon: at 0.01 the ship's edge cells flicker, at 0.08 they
sit still while the water keeps three quarters of its sparkle.

## A flag in the breeze

`flag` displaces the cells inside a box vertically, by a wave running out along the
cloth. Travel grows with distance from `pole`, given as a fraction across the box, so
the hoist stays pinned and the fly end moves most:

```bash
.venv/bin/python -m asciibg.cli seafront.png \
  --effect "flag:left=0.925,right=1.0,top=0.044,bottom=0.20,amplitude=1.5,waves=1.5,cycles=5"
```

Each column samples the *original* field rather than the running frame, which is what
lets a cell that moves away show the sky behind it instead of a smear of itself. Give
the box a row of margin above and below the cloth for the same reason.

`amplitude` is in cells, and the useful range is narrower than it looks. Below about
1.2 the shift rounds to zero for every column but the outermost, so the flag pops
rather than ripples; much above 2 and a flag a handful of cells deep is being shaken,
not blown. Find the box with the picture's own colours: a flag is the reddest thing
in a seascape, so `rgb[..., 0] - (rgb[..., 1] + rgb[..., 2]) / 2` locates it in a
line.

## True colour: lift the mids, do not clip them

`--clip-low` is for a subject the background is burying. Reach for it and the picture
goes dark, because everything below the percentile is gone, the subject with it. To
keep a picture's own colours, leave the clip at its default and lift the mid tones
instead, with `--gamma` *below* 1:

```bash
.venv/bin/python -m asciibg.cli seafront.png --clip-high 99 \
  --gamma 0.8 --value-boost 0.3 --saturation 1.1
```

`--ramp block` is truer still, because `█` and `▓` fill the cell and carry the colour
where `.` and `:` leave most of it black. It costs the ASCII texture, and the ramp is
5 levels rather than 10, so pair it with a lower `--gamma` or the mid tones collapse.

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

## Looping, for scenes

The same rule as the effects, and it is what the catalogue is built around: anything
that moves advances a whole number of wraps per loop, and anything random comes from
a table indexed by frame number. `loop_noise` is built from sines whose time
frequencies are whole numbers for the same reason - true value noise would need a
tileable lattice to manage it. Every scene in the catalogue is tested for this.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

47 tests. The scene suite runs the whole catalogue through the same checks: each one
loops, stays in 0..1, lights something, and never names a character outside its
atlas.
