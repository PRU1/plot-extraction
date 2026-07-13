"""Tests for the (working) classical-CV axis stage. Pure offline."""

import numpy as np
import pytest

from plot2spec.axes import detect_axis_lines, locate_ticks, match_ticks
from plot2spec.config import Config

from synthetic import make_spectrum_figure

TOL_PX = 3.0


@pytest.fixture(scope="module")
def synth():
    return make_spectrum_figure(seed=1)


def test_axis_lines_found(synth):
    x_axis_row, y_axis_col, plot_bbox = detect_axis_lines(synth.img_bgr)
    assert abs(x_axis_row - synth.x_axis_row) <= TOL_PX
    assert abs(y_axis_col - synth.y_axis_col) <= TOL_PX
    assert plot_bbox.width > 0.5 * synth.img_bgr.shape[1]
    assert plot_bbox.height > 0.4 * synth.img_bgr.shape[0]


def test_ticks_located(synth):
    x_axis_row, y_axis_col, plot_bbox = detect_axis_lines(synth.img_bgr)
    found = locate_ticks(synth.img_bgr, x_axis_row, y_axis_col, plot_bbox, "x")
    # every true tick should have a detection within tolerance
    for true_px in synth.xtick_px:
        assert min(abs(f - true_px) for f in found) <= TOL_PX, (
            f"tick at {true_px:.1f}px not found; detections: {found}"
        )
    # and not wildly many spurious detections
    assert len(found) <= len(synth.xtick_px) + 2


def test_match_ticks_equal_counts():
    ticks = match_ticks([10.0, 20.0, 30.0], [100.0, 200.0, 300.0])
    assert [(t.px, t.value) for t in ticks] == [(10.0, 100.0), (20.0, 200.0), (30.0, 300.0)]


def test_match_ticks_mismatch_keeps_pixels():
    ticks = match_ticks([10.0, 20.0, 30.0], [100.0, 200.0])
    assert len(ticks) == 3
    assert ticks[2].value is None or isinstance(ticks[2].value, float)
