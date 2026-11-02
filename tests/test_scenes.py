"""Scenes: every one must loop, stay in range, and only use glyphs it declared."""

from __future__ import annotations

import unittest

import numpy as np

from asciibg.check import LoopCheck
from asciibg.piece import Context, Meta, resolve_chars
from asciibg.prompt import scene_from_prompt
from asciibg.ramps import ramp
from asciibg.render import build_atlas, render_frame, sort_by_ink
from asciibg.scenes import PALETTES, SCENES, Scene, gradient, loop_noise, parse_scene

COLS, ROWS, FPS, LOOP = 48, 16, 8, 1.5


def build(scene: Scene):
    base = ramp(scene.ramp_name)
    extra = "".join(c for c in scene.charset_extra if c not in base)
    atlas = sort_by_ink(build_atlas(base + extra, 6, 12), limit=len(base))
    meta = Meta(type(scene).__name__.lower(), COLS, ROWS, FPS, LOOP,
                atlas.charset, ramp_size=len(base))
    scene.setup(meta)
    return atlas, meta


class TestEveryScene(unittest.TestCase):
    """Run the whole catalogue through the same checks."""

    def test_every_scene_loops(self):
        for name, cls in sorted(SCENES.items()):
            with self.subTest(scene=name):
                scene = cls()
                atlas, meta = build(scene)
                check = LoopCheck()
                for i in range(meta.frame_count):
                    cells = scene.render(Context(meta=meta, frame=i))
                    check.add(render_frame(
                        resolve_chars(cells, meta.ramp_len, atlas.size), cells.rgb, atlas
                    ))
                self.assertTrue(check.ok(), f"{name}: {check.report()}")

    def test_every_scene_stays_in_range(self):
        for name, cls in sorted(SCENES.items()):
            with self.subTest(scene=name):
                scene = cls()
                atlas, meta = build(scene)
                cells = scene.render(Context(meta=meta, frame=meta.frame_count // 3))
                self.assertEqual(cells.level.shape, (ROWS, COLS))
                self.assertGreaterEqual(float(cells.level.min()), 0.0)
                self.assertLessEqual(float(cells.level.max()), 1.0)
                self.assertGreaterEqual(float(cells.rgb.min()), 0.0)
                self.assertLessEqual(float(cells.rgb.max()), 1.0)

    def test_every_scene_lights_something(self):
        for name, cls in sorted(SCENES.items()):
            with self.subTest(scene=name):
                scene = cls()
                atlas, meta = build(scene)
                lit = max(
                    float((scene.render(Context(meta=meta, frame=i)).level > 0.05).mean())
                    for i in range(0, meta.frame_count, 3)
                )
                self.assertGreater(lit, 0.01, f"{name} renders almost nothing")

    def test_character_overrides_stay_inside_the_atlas(self):
        for name, cls in sorted(SCENES.items()):
            with self.subTest(scene=name):
                scene = cls()
                atlas, meta = build(scene)
                for i in range(0, meta.frame_count, 3):
                    cells = scene.render(Context(meta=meta, frame=i))
                    chars = resolve_chars(cells, meta.ramp_len, atlas.size)
                    self.assertGreaterEqual(int(chars.min()), 0)
                    self.assertLess(int(chars.max()), atlas.size)

    def test_scenes_without_extras_stay_inside_the_ramp(self):
        for name, cls in sorted(SCENES.items()):
            if cls.charset_extra:
                continue
            with self.subTest(scene=name):
                scene = cls()
                atlas, meta = build(scene)
                cells = scene.render(Context(meta=meta, frame=1))
                self.assertLess(
                    int(resolve_chars(cells, meta.ramp_len, atlas.size).max()), meta.ramp_len
                )


class TestRampChoice(unittest.TestCase):
    def test_continuous_scenes_avoid_the_shade_ramp(self):
        """Smooth fields drawn with `-`, `=` and `:` read as horizontal stripes."""
        for name in ("nebula", "waves"):
            self.assertEqual(SCENES[name].ramp_name, "block", name)

    def test_extras_sit_after_the_ramp_so_sorting_cannot_move_them(self):
        scene = SCENES["matrix"]()
        atlas, meta = build(scene)
        # Sorted by ink, so the order differs from the written ramp - but the same
        # characters must still be the ones brightness quantises into.
        self.assertEqual(sorted(atlas.charset[: meta.ramp_len]), sorted(ramp("shade")))
        for ch in scene.charset_extra:
            self.assertGreaterEqual(atlas.charset.index(ch), meta.ramp_len)


class TestPrimitives(unittest.TestCase):
    def test_loop_noise_is_bounded(self):
        x = np.linspace(0, 1.7, 40, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, 20, dtype=np.float32)[:, None]
        n = loop_noise(x, y, 0.3)
        self.assertGreaterEqual(float(n.min()), 0.0)
        self.assertLessEqual(float(n.max()), 1.0)

    def test_loop_noise_returns_to_its_start(self):
        x = np.linspace(0, 1.7, 40, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, 20, dtype=np.float32)[:, None]
        np.testing.assert_allclose(loop_noise(x, y, 0.0), loop_noise(x, y, 1.0), atol=1e-4)

    def test_loop_noise_varies_on_both_axes(self):
        """Drawing kx and ky separately left octaves axis-aligned, which banded."""
        x = np.linspace(0, 1.7, 64, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, 64, dtype=np.float32)[:, None]
        n = loop_noise(x, y, 0.2, octaves=3)
        across = float(np.abs(np.diff(n, axis=1)).mean())
        down = float(np.abs(np.diff(n, axis=0)).mean())
        self.assertGreater(across, down / 4.0)
        self.assertGreater(down, across / 4.0)

    def test_gradient_hits_its_endpoints(self):
        stops = PALETTES["ice"]
        t = np.array([[0.0, 1.0]], dtype=np.float32)
        out = gradient(t, stops)
        np.testing.assert_allclose(out[0, 0], stops[0][1], atol=1e-5)
        np.testing.assert_allclose(out[0, 1], stops[-1][1], atol=1e-5)


class TestSpecs(unittest.TestCase):
    def test_parses_numbers_and_palette(self):
        scene = parse_scene("matrix:density=0.7,palette=amber")
        self.assertEqual((scene.density, scene.palette), (0.7, "amber"))

    def test_rejects_unknown_scene(self):
        with self.assertRaises(ValueError):
            parse_scene("volcano")


class TestPrompt(unittest.TestCase):
    def test_picks_the_obvious_scene(self):
        for text, expected in [
            ("heavy rain at night", "rain"),
            ("slow purple nebula", "nebula"),
            ("green terminal code falling", "matrix"),
            ("a calm blue ocean", "waves"),
            ("retro synthwave grid", "grid"),
            ("distant stars in deep space", "starfield"),
        ]:
            with self.subTest(prompt=text):
                scene, why = scene_from_prompt(text)
                self.assertEqual(type(scene).__name__.lower(), expected, why)

    def test_colour_words_set_the_palette(self):
        self.assertEqual(scene_from_prompt("amber rain")[0].palette, "amber")
        self.assertEqual(scene_from_prompt("icy blue rain")[0].palette, "ice")

    def test_pace_and_density_words_tune_it(self):
        self.assertGreater(scene_from_prompt("fast rain")[0].speed,
                           scene_from_prompt("slow rain")[0].speed)
        self.assertGreater(scene_from_prompt("heavy rain")[0].density,
                           scene_from_prompt("sparse rain")[0].density)

    def test_falls_back_when_nothing_matches(self):
        scene, why = scene_from_prompt("a cat playing the piano")
        self.assertIn("nothing matched", why)
        self.assertIn(type(scene).__name__.lower(), SCENES)


if __name__ == "__main__":
    unittest.main()
