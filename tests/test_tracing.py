"""Tests for YOUR curve tracing. Skip until implemented."""

import numpy as np

from plot2spec.models import BBox, CurveMask
from plot2spec.tracing import trace_curves

from conftest import run_stub
from synthetic import make_crossing_mask, make_spectrum_figure


def _score_against_truth(trace, truth):
    """Mean |row error| of a trace vs. a ground-truth (col, row) trajectory."""
    truth_by_col = {int(round(c)): r for c, r in truth}
    errs = [abs(r - truth_by_col[int(round(c))])
            for c, r in trace.points_px if int(round(c)) in truth_by_col]
    return np.mean(errs) if errs else np.inf


def test_two_same_color_curves_crossing():
    """The canonical case: same color, one crossing. Identity must be
    preserved through the crossing (smoothness term doing its job)."""
    mask, truths = make_crossing_mask()
    cm = CurveMask(mask=mask, color_rgb=(0, 0, 0), n_expected_curves=2)
    traces = run_stub(trace_curves, [cm], BBox(0, 0, mask.shape[1], mask.shape[0]))
    assert len(traces) == 2
    # each truth matched by exactly one trace with small error
    errors = np.array([[_score_against_truth(t, tr) for tr in truths] for t in traces])
    best = errors.min(axis=1)
    assert (best < 3.0).all(), f"trace errors vs truth: {errors}"
    assert set(errors.argmin(axis=1)) == {0, 1}, "both traces matched the same truth curve"


def test_full_synthetic_figure():
    """Different-colored Lorentzian stacks from the synthetic generator."""
    from plot2spec.axes import detect_axis_lines
    from plot2spec.segmentation import segment_curves

    synth = make_spectrum_figure(n_curves=3, seed=3)
    _, _, plot_bbox = detect_axis_lines(synth.img_bgr)
    masks = segment_curves(synth.img_bgr, plot_bbox, vision=None)
    traces = run_stub(trace_curves, masks, plot_bbox)
    assert len(traces) == 3
    for truth in synth.curves_px:
        assert min(_score_against_truth(t, truth) for t in traces) < 3.0
