"""Tests for the (working) classical segmentation. Pure offline (no priors)."""

import numpy as np
import pytest

from plot2spec.axes import detect_axis_lines
from plot2spec.segmentation import segment_curves

from synthetic import make_spectrum_figure


@pytest.fixture(scope="module")
def synth():
    return make_spectrum_figure(n_curves=3, seed=2)


def _color_dist(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def test_one_mask_per_color(synth):
    _, _, plot_bbox = detect_axis_lines(synth.img_bgr)
    masks = segment_curves(synth.img_bgr, plot_bbox, vision=None)
    # every plotted color should be represented by some mask
    for true_rgb in synth.colors_rgb:
        assert any(_color_dist(m.color_rgb, true_rgb) < 90 for m in masks), (
            f"no mask matches curve color {true_rgb}; "
            f"mask colors: {[m.color_rgb for m in masks]}"
        )


def test_mask_pixels_lie_on_curves(synth):
    _, _, plot_bbox = detect_axis_lines(synth.img_bgr)
    masks = segment_curves(synth.img_bgr, plot_bbox, vision=None)
    for true_rgb, true_px in zip(synth.colors_rgb, synth.curves_px):
        mask = min(masks, key=lambda m: _color_dist(m.color_rgb, true_rgb))
        rows, cols = np.nonzero(mask.mask)
        assert len(rows) > 100
        # for each masked pixel, distance to the true trajectory at that column
        truth_by_col = {}
        for c, r in true_px:
            truth_by_col.setdefault(int(round(c)), []).append(r)
        errs = [
            min(abs(r - tr) for tr in truth_by_col[c])
            for r, c in zip(rows, cols)
            if c in truth_by_col
        ]
        assert np.median(errs) <= 2.5, f"median mask deviation {np.median(errs):.1f}px"
