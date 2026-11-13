# asciibg

Seamlessly looping ASCII animations, exported as an MP4 for video-call backgrounds
and a GIF for the web. Start from your own photo, from one of six procedural scenes,
or from a sentence describing what you want.

The look follows [ascii.rest](https://ascii.rest): the art is quantised once into a
fixed grid of characters, and motion is a layer over the top. Every loop closes
exactly — frame *N* is never rendered, because it *is* frame 0.

```bash
asciibg --scene matrix:density=0.7,palette=amber -o out
asciibg --prompt "slow heavy snow on a cold blue night" -o out
asciibg photo.jpg --loop 12 --center-dim 0.5 -o out
```

## Install

```bash
pip install git+https://github.com/MttJns/asciibg.git
```

Or from a clone, for hacking on it:

```bash
git clone https://github.com/MttJns/asciibg.git
cd asciibg
python3 -m venv .venv && .venv/bin/pip install -e .
```

Python 3.9 or newer. Two things have to be on the machine already:

**ffmpeg**, on `PATH`. Frames are piped straight into it.

```bash
brew install ffmpeg                       # macOS
sudo apt-get install ffmpeg               # Debian, Ubuntu
winget install Gyan.FFmpeg                # Windows
```

**A monospace font.** Nothing is downloaded; `asciibg` searches the system font
directories and takes the best fixed-pitch face it finds — Menlo, SF Mono or Andale
Mono on macOS, DejaVu Sans Mono, Liberation Mono, Noto Sans Mono or Ubuntu Mono on
Linux, Consolas, Lucida Console or Courier New on Windows. Point it somewhere else
with `--font /path/to/Mono.ttf`. It will refuse a proportional font rather than
silently shear the grid, because every cell is exactly one advance wide.

## Use

```bash
asciibg photo.jpg -o out
```

Writes four files:

| file | what it is |
|---|---|
| `out/photo.mp4` | 1920x1080, H.264, yuv420p, faststart — the background itself |
| `out/photo.gif` | the same loop, for the web |
| `out/photo.png` | the first frame, usable as a still background |
| `out/photo-sheet.png` | a 3x3 contact sheet with the speaker's safe zone marked |

The CLI exits non-zero if the loop check fails, so it drops into a script or a CI job
as-is. `asciibg --help` lists every flag; `asciibg --list-scenes` lists the catalogue.

`--still N` renders one frame to the PNG and stops, in about half a second and with no
ffmpeg needed. Judge framing and tone there rather than waiting out a video. For
measuring a picture before animating it — where things are, which way a surface slopes,
whether the motion you set up actually travels — see `tools/probe.py` and the recipe in
[CONTRIBUTING.md](CONTRIBUTING.md).

```bash
asciibg photo.jpg \
  --loop 12 --ramp dot --center-dim 0.5 \
  --effect "glint:period=5,strength=0.6" \
  --effect "scan:period=9" \
  --effect "drift:amplitude=0.08"
```

## Scenes: backgrounds without an image

Scenes are computed per frame rather than quantised from a picture, the way
ascii.rest's own scenes are. Pick one directly, or describe what you want:

```bash
asciibg --scene waves:palette=ember -o out
asciibg --prompt "retro synthwave grid" -o out
asciibg --list-scenes
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

`--prompt` selects and tunes a scene; it does not generate one, and it calls no
network service. A fixed catalogue that is known to loop and known to read as a
background is worth more than free-form generation that may do neither. Colour words
set the palette, pace words (`fast`/`slow`) and quantity words (`heavy`/`sparse`)
turn the knobs, and a prompt matching nothing falls back to `nebula` and says so.

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
   that scrambles the characters it passes, `drift` is a slow brightness wave,
   `twinkle` sparkles isolated dim points such as a starfield, `sparkle` winks the
   bright ones such as sun on water, and `flag` ripples a flag on its pole. The grid
   itself never changes.
4. **Render.** Every character is rasterised once at the exact cell size into an
   alpha mask, then blitted tinted by each cell's colour. Nothing is ever resampled,
   which is what keeps the glyph edges crisp.
5. **Encode.** Raw frames pipe straight into ffmpeg; the GIF is made from the video.

## Looping

Every effect and every scene is a function of the loop phase alone, built from
periodic terms or from tables indexed by frame number. Anything that moves advances a
whole number of wraps per loop, and anything random comes from a frame-indexed table.
`loop_noise` is built from sines whose time frequencies are whole numbers for the
same reason — true value noise would need a tileable lattice to manage it.

Frame count is exactly `loop * fps`, and frame `N` is never rendered because it *is*
frame 0. Each render reports a check comparing the last-to-first step against the
90th percentile of the ordinary frame-to-frame steps. The percentile matters: sparse
motion, such as a small sprite on a still background, leaves most consecutive frames
identical, so a median baseline is zero and calls an exact loop a break.

Every scene in the catalogue is tested for this, and so is every effect.

## Tuning a photo

### True colour: lift the mids, do not clip them

`--clip-low` is for a subject the background is burying. Reach for it and the picture
goes dark, because everything below the percentile is gone, the subject with it. To
keep a picture's own colours, leave the clip at its default and lift the mid tones
instead, with `--gamma` *below* 1:

```bash
asciibg photo.jpg --clip-high 99 --gamma 0.8 --value-boost 0.3 --saturation 1.1
```

`--ramp block` is truer still, because `█` and `▓` fill the cell and carry the colour
where `.` and `:` leave most of it black. It costs the ASCII texture, and the ramp is
5 levels rather than 10, so pair it with a lower `--gamma` or the mid tones collapse.

### Subjects on a dark background

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
asciibg spaceship.jpg --loop 10 \
  --cols 240 --rows 60 --zoom 0.45 --offset-x -0.9 --offset-y 0.45 \
  --clip-low 70 --clip-high 99.5 --gamma 0.7 --value-boost 0.85 --saturation 1.4 \
  --effect "twinkle:amplitude=0.45,density=0.6"
```

### Bright highlights: `sparkle` rather than `twinkle`

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
asciibg seafront.jpg --loop 10 \
  --clip-low 68 --clip-high 99 --value-boost 0.7 \
  --effect "sparkle:top=0.22,bottom=0.72,density=0.75,dim=0.3,isolation=0.08"
```

Raise `isolation` to keep it off a solid bright mass that happens to sit in the band,
such as a hull on the horizon: at 0.01 the ship's edge cells flicker, at 0.08 they
sit still while the water keeps three quarters of its sparkle.

### A flag in the breeze

`flag` displaces the cells inside a box vertically, by a wave running out along the
cloth. Travel grows with distance from `pole`, given as a fraction across the box, so
the hoist stays pinned and the fly end moves most:

```bash
asciibg seafront.jpg \
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

### Barrels down a girder

`roll` sends a barrel the picture already contains rolling along a girder. Nothing is
drawn: the cells inside the home box are lifted out of the quantised original and
carried, so what rolls is the artwork's own barrel, with the colour and shape the
picture gave it. The box is blanked where the barrel was, because a barrel that has
rolled away is no longer where it started.

```bash
asciibg arcade.png --cols 240 --rows 60 --offset-y -1 \
  --effect "roll:left=0.308,right=0.354,top=0.583,bottom=0.650,\
x0=0,y0=0.667,x1=0.925,y1=0.700,cycles=6"
```

`left`, `right`, `top` and `bottom` are the home box and the track runs from
(`x0`, `y0`) to (`x1`, `y1`), all in fractions of the grid. Where the barrel sits on
the track, and how high above it, are read from the box rather than given, so **frame
0 is the untouched picture** — a useful thing to assert in a test, and the reason one
effect instance carries one barrel.

Finding the barrels is a measurement. They have a colour of their own, so mask for it
and take connected components: on the Donkey Kong 25m screen,
`(r > 0.45) & (g > 0.20) & (g < r*0.85) & (b < g*0.8)` with a two-column gap
tolerance separates the four rolling barrels from the stack, from the oil drum and
from Donkey Kong himself, and each component's bounding box is the home box. Use the
box tight: every cell inside these four is barrel, so `floor` never has to discard
anything, but a looser box would drag a rectangle of background along the girder.

Finding a track is the same kind of measurement. A girder has its own colour too —
`(r > 0.35) & (b > 0.25) & (g < r * 0.7)` picks out magenta beams — so read the
topmost masked row per column at each end of its span. The sign of the slope is also
the direction of travel: barrels roll downhill, so swap the endpoints when the right
end is the higher one.

Keep the travel per frame near a cell or two. Motion is only motion if consecutive
frames overlap, and the loop check cannot see the difference: a barrel jumping a
sixth of the screen per frame strobes between a handful of fixed positions, and the
check still calls it seamless at a ratio of 1.00, because every frame is as unlike
its neighbour as the wrap is.

### Shrinking a subject: zoom below 1

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

## Making it work as a video-call background

- Zoom takes MP4/MOV video and JPG/PNG stills, at up to 1920x1080. Check your own
  client's limits; other apps differ.
- Your head and shoulders cover roughly the centre of the frame. `--center-dim 0.5`
  fades that area so your face reads against it; the contact sheet marks the zone.
- "Mirror my video" flips your self-view, so lettering reads backwards to you but
  correctly to everyone else. Prefer art without text.

### Keeping the subject out from behind your head

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

For one spaceship photo, centred put 50% of the ship behind the speaker's head.
Sweeping zoom and offset found `--zoom 1.5 --offset-x 0.9 --offset-y 1.0`, which
drops it to 3.7% by lifting the ship into a banner across the top, at the cost of the
nose running off the left edge.

## Geometry

The grid must divide the frame exactly, so cells land on whole pixels. The default
160x45 into 1920x1080 gives 12x24 px cells, and Menlo at size 20 has an advance of
exactly 12 px and a height of exactly 24. Other exact fits: 192x54, 240x60, 128x36.
A grid that does not divide is rejected with the arithmetic and some suggestions,
rather than quietly rounded.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m unittest discover -s tests
```

62 tests, no network and no fixtures on disk. The scene suite runs the whole
catalogue through the same checks — each one loops, stays in 0..1, lights something,
and never names a character outside its atlas — so a seventh scene inherits them.

CI runs the tests on Python 3.9 through 3.13 on Linux, and once each on macOS and
Windows, then builds a wheel and renders a real loop from it.

`PLAN.md` holds the original design notes and the work deliberately left undone
(edge-detected "logo" style, k-means palettes, preset files, depth parallax).

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT. See [LICENSE](LICENSE).
