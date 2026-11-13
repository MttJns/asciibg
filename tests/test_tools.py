"""The measuring tools, and the one-frame preview they are used with."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from asciibg.cli import main
from asciibg.effects import Roll
from asciibg.piece import Field, Meta
from tools.probe import SMOOTH_TRAVEL, blobs, hue_mask, motion, ridge


def solid(rows: int, cols: int) -> Field:
    return Field(level=np.zeros((rows, cols), dtype=np.float32),
                 rgb=np.zeros((rows, cols, 3), dtype=np.float32))


class TestHueMask(unittest.TestCase):
    def setUp(self):
        self.field = solid(10, 10)
        self.field.level[:] = 0.9
        self.field.rgb[:, 0] = (1.0, 0.5, 0.0)      # tan, hue 30
        self.field.rgb[:, 1] = (0.9, 0.0, 0.7)      # magenta, hue 313
        self.field.rgb[:, 2] = (0.7, 0.7, 0.7)      # grey, no hue at all

    def test_it_picks_the_hue_asked_for(self):
        mask = hue_mask(self.field, 30.0, 20.0, 0.25)
        self.assertTrue(mask[:, 0].all())
        self.assertFalse(mask[:, 1].any())

    def test_grey_belongs_to_no_hue(self):
        # Grey has a hue of sorts once you divide by a chroma of nearly zero, so it
        # has to be excluded by its saturation or it matches whatever you ask for.
        for hue in (0.0, 30.0, 120.0, 313.0):
            self.assertFalse(hue_mask(self.field, hue, 40.0, 0.25)[:, 2].any())

    def test_dark_cells_are_not_counted(self):
        self.field.level[:, 0] = 0.1
        self.assertFalse(hue_mask(self.field, 30.0, 20.0, 0.25)[:, 0].any())


class TestBlobs(unittest.TestCase):
    def test_separate_things_come_back_separately(self):
        mask = np.zeros((20, 40), dtype=bool)
        mask[4:8, 2:8] = True
        mask[4:8, 20:26] = True
        found = blobs(mask, gap=0, min_cells=8)
        self.assertEqual(len(found), 2)
        self.assertEqual({f[3:] for f in found}, {(2, 7), (20, 25)})

    def test_a_gap_bridges_a_break_and_no_gap_does_not(self):
        mask = np.zeros((20, 40), dtype=bool)
        mask[4:8, 2:8] = True
        mask[4:8, 9:15] = True      # one blank column between the halves
        self.assertEqual(len(blobs(mask, gap=0, min_cells=8)), 2)
        self.assertEqual(len(blobs(mask, gap=1, min_cells=8)), 1)

    def test_specks_below_the_floor_are_dropped(self):
        mask = np.zeros((20, 40), dtype=bool)
        mask[4:8, 2:8] = True
        mask[0, 30] = True
        self.assertEqual(len(blobs(mask, gap=0, min_cells=8)), 1)


class TestRidge(unittest.TestCase):
    def _sloped(self, rise: int) -> np.ndarray:
        mask = np.zeros((20, 40), dtype=bool)
        for col in range(40):
            top = 8 + round(rise * col / 39.0)
            mask[top:top + 3, col] = True     # a beam three rows deep
        return mask

    def test_it_reads_the_top_of_the_beam_not_its_middle(self):
        # A line fitted through the whole beam sits a row inside it; a barrel rests
        # on the top.
        self.assertEqual(ridge(self._sloped(0), (0, 19)), (0, 8, 39, 8))

    def test_it_reports_both_ends(self):
        self.assertEqual(ridge(self._sloped(6), (0, 19)), (0, 8, 39, 14))

    def test_too_little_surface_is_no_surface(self):
        mask = np.zeros((20, 40), dtype=bool)
        mask[5, :4] = True
        self.assertIsNone(ridge(mask, (0, 19)))


class TestMotion(unittest.TestCase):
    """The check the loop check cannot make: do consecutive frames still overlap?"""

    def setUp(self):
        self.field = solid(20, 60)
        self.field.level[6:10, 4:10] = 0.9
        self.field.rgb[6:10, 4:10] = (1.0, 0.5, 0.0)
        self.home = {"left": 4 / 60, "right": 10 / 60, "top": 6 / 20, "bottom": 10 / 20}
        self.track = {"x0": 0.0, "y0": 0.6, "x1": 1.0, "y1": 0.6}

    def _measure(self, cycles):
        # Enough frames that one crossing is many steps, or every step is a jump and
        # there is nothing to tell a smooth pass from a strobing one.
        meta = Meta("t", 60, 20, 60, 1.0, " .@")
        effect = Roll(cycles=cycles, **self.home, **self.track)
        return motion(self.field, meta, [effect])[0]

    def test_travel_is_a_share_of_the_thing_s_own_width(self):
        distinct, step, width = self._measure(1)
        self.assertEqual(width, 6)         # the barrel is six cells across
        self.assertLess(step / width, SMOOTH_TRAVEL)

    def test_it_catches_motion_that_outruns_the_frame_rate(self):
        slow, fast = self._measure(1), self._measure(10)
        self.assertGreater(fast[1] / fast[2], slow[1] / slow[2])
        self.assertGreater(fast[1] / fast[2], SMOOTH_TRAVEL)
        # And the giveaway the loop check misses: far fewer frames than the loop has.
        self.assertLess(fast[0], slow[0])


class TestStill(unittest.TestCase):
    def test_it_writes_one_frame_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "src.png")
            Image.fromarray(np.full((180, 320, 3), 120, dtype=np.uint8)).save(path)
            err, out_text = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out_text):
                code = main([path, "-o", out, "-n", "one", "--cols", "32", "--rows", "9",
                             "--width", "320", "--height", "180", "--still", "7"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(os.path.join(out, "one.png")))
            for unwanted in ("one.mp4", "one.gif", "one-sheet.png"):
                self.assertFalse(os.path.exists(os.path.join(out, unwanted)), unwanted)


if __name__ == "__main__":
    unittest.main()
