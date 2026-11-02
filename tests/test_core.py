"""Core behaviour: grid fit, ramp ordering, quantising, and loop seamlessness."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from asciibg.check import LoopCheck
from asciibg.effects import Drift, Glint, Scan, Twinkle, parse_effect, parse_effects
from asciibg.piece import Context, Meta, cells_from_field, resolve_chars
from asciibg.quantize import (
    _padding_mask, cover_crop, dilate, extend_sky, field_from_image, frame_source,
)
from asciibg.ramps import ramp
from asciibg.render import build_atlas, render_frame, sort_by_ink

COLS, ROWS, CELL_W, CELL_H, FPS, LOOP = 40, 12, 12, 24, 12, 2.0


def make_image(tmp: str) -> str:
    yy, xx = np.mgrid[0:180, 0:320]
    sun = np.exp(-(((xx - 220) / 40.0) ** 2 + ((yy - 60) / 40.0) ** 2))
    base = np.stack([0.2 + 0.4 * (1 - yy / 180), 0.3 * (1 - yy / 180), 0.5 * (1 - yy / 180)], -1)
    img = np.clip(base + sun[..., None] * np.array([0.9, 0.7, 0.3]), 0, 1)
    Image.fromarray((img * 255).astype("uint8")).save(tmp)
    return tmp


class TestMeta(unittest.TestCase):
    def test_frame_count_is_exact(self):
        self.assertEqual(Meta("x", 10, 4, 30, 12.0, " .@").frame_count, 360)
        self.assertEqual(Meta("x", 10, 4, 24, 2.5, " .@").frame_count, 60)


class TestAtlas(unittest.TestCase):
    def setUp(self):
        self.atlas = build_atlas(ramp("shade"), CELL_W, CELL_H)

    def test_masks_are_cell_sized(self):
        self.assertEqual(self.atlas.masks.shape, (self.atlas.size, CELL_H, CELL_W))

    def test_space_draws_nothing(self):
        self.assertEqual(self.atlas.masks[0].sum(), 0)

    def test_sorting_makes_ink_monotonic(self):
        ink = sort_by_ink(self.atlas).masks.reshape(self.atlas.size, -1).mean(axis=1)
        self.assertTrue(np.all(np.diff(ink) >= 0), ink)

    def test_unsorted_menlo_ramp_is_not_monotonic(self):
        ink = self.atlas.masks.reshape(self.atlas.size, -1).mean(axis=1)
        self.assertFalse(np.all(np.diff(ink) >= 0))


class TestQuantize(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.tmp.close()
        self.path = make_image(self.tmp.name)

    def test_field_shape_and_range(self):
        field = field_from_image(self.path, COLS, ROWS)
        self.assertEqual(field.level.shape, (ROWS, COLS))
        self.assertEqual(field.rgb.shape, (ROWS, COLS, 3))
        self.assertGreaterEqual(field.level.min(), 0.0)
        self.assertLessEqual(field.level.max(), 1.0)

    def test_cover_crop_hits_target_aspect(self):
        wide = cover_crop(Image.new("RGB", (400, 100)), 16 / 9)
        tall = cover_crop(Image.new("RGB", (100, 400)), 16 / 9)
        for img in (wide, tall):
            self.assertAlmostEqual(img.size[0] / img.size[1], 16 / 9, places=1)

    def test_value_boost_lifts_colour_without_changing_hue(self):
        plain = field_from_image(self.path, COLS, ROWS)
        lifted = field_from_image(self.path, COLS, ROWS, value_boost=1.0)
        self.assertGreater(lifted.rgb.max(axis=-1).mean(), plain.rgb.max(axis=-1).mean())
        bright = plain.rgb.max(axis=-1) > 0.1
        ratio_plain = plain.rgb[bright] / plain.rgb[bright].max(axis=-1, keepdims=True)
        ratio_lifted = lifted.rgb[bright] / lifted.rgb[bright].max(axis=-1, keepdims=True)
        np.testing.assert_allclose(ratio_plain, ratio_lifted, atol=1e-4)

    def test_raising_the_low_clip_darkens_the_background(self):
        plain = field_from_image(self.path, COLS, ROWS, clip=(2.0, 98.0))
        lifted = field_from_image(self.path, COLS, ROWS, clip=(70.0, 99.5))
        self.assertGreater((lifted.level == 0.0).sum(), (plain.level == 0.0).sum())

    def test_zoom_and_offset_move_the_crop(self):
        img = Image.new("RGB", (400, 400))
        full = cover_crop(img, 16 / 9)
        zoomed = cover_crop(img, 16 / 9, zoom=2.0)
        self.assertLess(zoomed.size[0], full.size[0])
        self.assertAlmostEqual(zoomed.size[0] / zoomed.size[1], 16 / 9, places=1)
        left = cover_crop(img, 16 / 9, zoom=2.0, offset=(-1.0, 0.0))
        right = cover_crop(img, 16 / 9, zoom=2.0, offset=(1.0, 0.0))
        self.assertNotEqual(left.size, (0, 0))
        self.assertEqual(left.size, right.size)

    def test_center_dim_darkens_the_middle(self):
        plain = field_from_image(self.path, COLS, ROWS)
        dimmed = field_from_image(self.path, COLS, ROWS, center_dim=0.8)
        mid = (slice(ROWS // 3, 2 * ROWS // 3), slice(COLS // 3, 2 * COLS // 3))
        self.assertLess(dimmed.level[mid].mean(), plain.level[mid].mean())


class TestZoomOut(unittest.TestCase):
    """Zooming below 1 pads around the image, and the padding has to pass for sky."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.tmp.close()
        self.path = make_image(self.tmp.name)
        self.z, self.off = 0.5, (-0.9, 0.0)
        img = Image.open(self.path)
        _, self.content = frame_source(img, 16 / 9, zoom=self.z, offset=self.off)
        self.padding = _padding_mask(self.content, COLS, ROWS)

    def test_zoom_out_creates_padding(self):
        self.assertGreater(self.padding.mean(), 0.5)

    def test_negative_offset_puts_the_subject_right(self):
        x0, _, x1, _ = self.content
        self.assertGreater((x0 + x1) / 2, 0.5)

    def test_percentiles_ignore_the_padding(self):
        """Counting black padding drags the low percentile to zero, disabling the clip."""
        field = field_from_image(self.path, COLS, ROWS, zoom=self.z, offset=self.off,
                                 clip=(70.0, 99.5), sky_fill=False)
        real = ~self.padding
        self.assertLess((field.level[real] > 0.01).mean(), 0.5)

    def test_padding_matches_the_real_sky(self):
        field = field_from_image(self.path, COLS, ROWS, zoom=self.z, offset=self.off,
                                 clip=(70.0, 99.5))
        sky = (~self.padding) & ~dilate(field.level > 0.4, 2)
        self.assertGreater(sky.sum(), 10)
        self.assertAlmostEqual(
            float(field.level[sky].mean()), float(field.level[self.padding].mean()), delta=0.05
        )

    def test_sky_fill_can_be_turned_off(self):
        bare = field_from_image(self.path, COLS, ROWS, zoom=self.z, offset=self.off,
                                sky_fill=False)
        self.assertEqual(float(bare.level[self.padding].max()), 0.0)


class TestTwinkle(unittest.TestCase):
    """Twinkle must find the stars and leave a large bright subject alone."""

    def setUp(self):
        rows, cols = 24, 48
        self.level = np.zeros((rows, cols), dtype=np.float32)
        self.level[8:16, 4:20] = 0.9  # a bright subject, large and solid
        rng = np.random.default_rng(5)
        for y, x in zip(rng.integers(0, rows, 60), rng.integers(28, cols, 60)):
            self.level[y, x] = 0.2  # isolated points, as stars quantise to
        self.field = __import__("asciibg.piece", fromlist=["Field"]).Field(
            level=self.level, rgb=np.ones((rows, cols, 3), dtype=np.float32)
        )
        self.meta = Meta("t", cols, rows, 10, 1.0, " .-:+*=%#@")

    def _active(self):
        tw = Twinkle()
        active, _, _ = tw._stars(Context(meta=self.meta, frame=0, field=self.field), self.level)
        return active

    def test_picks_isolated_points(self):
        self.assertGreater(self._active().sum(), 0)

    def test_leaves_a_solid_bright_subject_alone(self):
        self.assertEqual(self._active()[8:16, 4:20].sum(), 0)

    def test_brightens_only_its_own_cells(self):
        active = self._active()
        cells = cells_from_field(self.field)
        Twinkle().apply(cells, Context(meta=self.meta, frame=3, field=self.field))
        lifted = cells.level > self.level
        self.assertTrue(np.all(~lifted | active))


class TestEffectSpecs(unittest.TestCase):
    def test_parses_name_and_arguments(self):
        effect = parse_effect("glint:period=4,strength=0.6")
        self.assertIsInstance(effect, Glint)
        self.assertEqual((effect.period, effect.strength), (4.0, 0.6))

    def test_rejects_unknown_effect(self):
        with self.assertRaises(ValueError):
            parse_effects(["kenburns"])


class TestLoopSeamlessness(unittest.TestCase):
    """The real guarantee: the last frame must join back onto the first."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.tmp.close()
        self.path = make_image(self.tmp.name)
        self.atlas = sort_by_ink(build_atlas(ramp("shade"), CELL_W, CELL_H))
        self.field = field_from_image(self.path, COLS, ROWS)
        self.meta = Meta("t", COLS, ROWS, FPS, LOOP, self.atlas.charset)

    def _run(self, effects):
        check = LoopCheck()
        for i in range(self.meta.frame_count):
            cells = cells_from_field(self.field)
            ctx = Context(meta=self.meta, frame=i, field=self.field)
            for effect in effects:
                effect.apply(cells, ctx)
            check.add(render_frame(resolve_chars(cells, self.atlas.size), cells.rgb, self.atlas))
        return check

    def test_glint_loops(self):
        self.assertTrue(self._run([Glint(period=1.0)]).ok(), self._run([Glint(period=1.0)]).report())

    def test_scan_loops(self):
        check = self._run([Scan(period=1.0)])
        self.assertTrue(check.ok(), check.report())

    def test_drift_loops(self):
        check = self._run([Drift()])
        self.assertTrue(check.ok(), check.report())

    def test_twinkle_loops(self):
        check = self._run([Twinkle(amplitude=0.5)])
        self.assertTrue(check.ok(), check.report())

    def test_all_effects_together_loop(self):
        check = self._run([Glint(period=1.0), Scan(period=2.0), Drift(), Twinkle()])
        self.assertTrue(check.ok(), check.report())

    def test_static_field_has_no_wrap_break(self):
        check = self._run([])
        self.assertTrue(check.ok(), check.report())


class TestRender(unittest.TestCase):
    def test_frame_shape_and_background(self):
        atlas = build_atlas(ramp("shade"), CELL_W, CELL_H)
        idx = np.zeros((ROWS, COLS), dtype=np.int16)  # all spaces
        rgb = np.ones((ROWS, COLS, 3), dtype=np.float32)
        frame = render_frame(idx, rgb, atlas, background=(7, 8, 9))
        self.assertEqual(frame.shape, (ROWS * CELL_H, COLS * CELL_W, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertTrue((frame == np.array([7, 8, 9], np.uint8)).all())


class TestCli(unittest.TestCase):
    def test_grid_must_divide_the_frame(self):
        from asciibg.cli import build_parser, _check_grid

        args = build_parser().parse_args(["x.png", "--cols", "77"])
        with self.assertRaises(SystemExit):
            _check_grid(args)

    def test_default_grid_divides_1080p(self):
        from asciibg.cli import build_parser, _check_grid

        self.assertEqual(_check_grid(build_parser().parse_args(["x.png"])), (12, 24))


if __name__ == "__main__":
    unittest.main()
