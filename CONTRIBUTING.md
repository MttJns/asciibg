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

## Adding an effect

Subclass `Effect` in `asciibg/effects.py` and add it to `EFFECTS`. `apply` mutates
one frame's `Cells` in place, and `parse_effect` turns every value in a CLI spec
into a float, so keep the parameters numeric.

The one hard rule is that the loop has to close. Build the motion out of periodic
terms in `ctx.phase`, use whole numbers of cycles per loop, and take any randomness
from a table indexed by frame number rather than from a fresh draw. Add a case to
`TestLoopSeamlessness`.

Two things worth knowing before you write a new one:

- Effects that *add* brightness do nothing to a cell already at the top of the ramp.
  That is why `sparkle` scales where `twinkle` adds.
- An effect that displaces cells should sample `ctx.field`, the quantised original,
  rather than the running frame. Sampling the running frame smears.

## House style

Match what is there: type hints, docstrings on public functions only, comments that
explain *why* rather than what. Measure claims about how something looks — most of
the parameter defaults in this repo came from counting cells across a loop, not from
eyeballing one frame, and the reasoning is in the commit messages.
