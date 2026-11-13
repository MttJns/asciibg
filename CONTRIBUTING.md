# Contributing

Issues and pull requests welcome.

## Running the tests

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m unittest discover -s tests
```

They need no network and no fixture files: images are generated in-memory, and
nothing shells out to ffmpeg. CI runs the same suite on Python 3.9 through 3.13 on
Linux and once each on macOS and Windows, then builds a wheel and renders a real
loop from it.

## Adding a scene

Subclass `Scene` in `asciibg/scenes.py`, give it `tags` so `--prompt` can find it,
and add it to the `SCENES` dict. The suite in `tests/test_scenes.py` picks it up
automatically and will hold it to the same contract as the rest: it must loop, stay
within 0..1, light something, and never name a character outside its atlas.

If it draws a smooth, continuous field, set `ramp_name = "block"`. The shade ramp's
`-`, `=` and `:` are all horizontal strokes, so a continuous field drawn with it
reads as stripes. If it needs characters a brightness ramp would not carry, list
them in `charset_extra`; they go into the atlas after the ramp, so brightness never
quantises into them.

## Animating a real picture

Every parameter of an effect is a fact about the picture: where a thing is, which way
its surface slopes, how far it may travel between frames. Guessing them costs renders,
so measure them. `tools/probe.py` takes the same flags as `asciibg` itself, which
means the framing you measure is the framing you render.

1. **Frame it on a still, not on a video.** `--still N` renders one frame to the PNG
   and stops. It takes about half a second where a full render takes half a minute,
   and it needs no ffmpeg, so iterate on framing and tone here.

2. **Find the things by their own colour.** `--find blobs --hue 25` boxes everything
   tan; `--hue 320` everything magenta. Each box is printed as
   `left=..,right=..,top=..,bottom=..`, ready to paste into an effect. Tighten
   `--spread` and `--gap` until the number of boxes matches what you can see: too
   generous a gap joins a barrel to the hand holding it.

3. **Find the surfaces the same way.** `--find ridge --hue 320 --band 36-46` reads the
   topmost masked row per column and prints `x0=..,y0=..,x1=..,y1=..`. It reports the
   *top* of a beam rather than a line fitted through it, because that is what a thing
   rests on, and it orders the endpoints downhill, because that is the way a thing
   rolls.

4. **Check the motion afterwards.** `--find motion --effect ...` reports how many
   distinct frames the loop really has and how far the moving thing travels per frame
   as a share of its own width. **The loop check cannot do this.** It asks whether the
   loop closes, not whether the frames in between join up, and it will call a strobe
   seamless at a ratio of 1.00.

## Things that have caught people out

| | |
|---|---|
| `--offset-x/y` is a fraction of the *slack*, not of the image, and is clamped to it | Four different values rendered the same frame before anyone noticed `-1.0` was what "flush to the top" meant |
| Crossing time is `loop / cycles`, and `cycles` is rounded to a whole number | A loop has to contain whole traversals. To change speed by a non-integer factor, change `--loop` |
| A glyph outside the ramp silently draws nothing | `ctx.meta.charset.find` returns -1. Worse, a test whose atlas is the ramp alone then passes on an empty frame, so assert `check.step_delta > 0` |
| Adding brightness does nothing at the top of the ramp | Why `sparkle` scales where `twinkle` adds |
| Displacing cells means sampling `ctx.field` | Sampling the running frame smears |

## Adding an effect

Subclass `Effect` in `asciibg/effects.py` and add it to `EFFECTS`. `apply` mutates
one frame's `Cells` in place, and `parse_effect` turns every value in a CLI spec
into a float, so keep the parameters numeric.

The one hard rule is that the loop has to close. Build the motion out of periodic
terms in `ctx.phase`, use whole numbers of cycles per loop, and take any randomness
from a table indexed by frame number rather than from a fresh draw. Add a case to
`TestLoopSeamlessness`.

Two things worth knowing before you write a new one:

- The table above is the short version; all of it applies to a new effect.
- An effect that moves part of the picture should lift it from `ctx.field` once and
  blank where it came from, as `roll` does. Deriving its starting place from the box
  rather than taking it as a parameter makes frame 0 the untouched picture, which a
  test can assert exactly.
- Measure the motion with `--find motion` before you believe it. `TestLoopSeamlessness`
  proves the loop closes and nothing more.

## House style

Match what is there: type hints, docstrings on public functions only, comments that
explain *why* rather than what. Measure claims about how something looks — most of
the parameter defaults in this repo came from counting cells across a loop, not from
eyeballing one frame, and the reasoning is in the commit messages.
