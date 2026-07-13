"""Tests for YOUR calibration math. They skip until you implement it,
then activate automatically (see conftest.run_stub)."""

import numpy as np
import pytest

from plot2spec.calibration import fit_axis_map
from plot2spec.models import TickMark

from conftest import run_stub


def ticks(pairs):
    return [TickMark(px=p, value=v) for p, v in pairs]


def test_linear_clean():
    # value = 2*px + 0  ->  ticks at px 100..400
    m = run_stub(fit_axis_map, ticks([(100, 200), (200, 400), (300, 600), (400, 800)]))
    assert m.kind == "linear"
    np.testing.assert_allclose(m.to_data(np.array([150.0, 250.0])), [300.0, 500.0], atol=1e-6)


def test_linear_noisy_pixels():
    rng = np.random.default_rng(0)
    px = np.array([100.0, 200, 300, 400, 500])
    v = 0.5 * px + 10
    m = run_stub(fit_axis_map, ticks(zip(px + rng.normal(0, 0.5, len(px)), v)))
    np.testing.assert_allclose(m.to_data(np.array([350.0])), [185.0], atol=1.5)


def test_single_ocr_outlier_rejected():
    # correct: v = 2*px; one gross OCR misread (800 -> 300)
    m = run_stub(fit_axis_map, ticks([(100, 200), (200, 400), (300, 600), (400, 300), (500, 1000)]))
    np.testing.assert_allclose(m.to_data(np.array([400.0])), [800.0], rtol=0.01)


def test_none_values_excluded():
    m = run_stub(fit_axis_map, ticks([(100, 200), (200, None), (300, 600), (400, 800)]))
    np.testing.assert_allclose(m.to_data(np.array([200.0])), [400.0], rtol=0.01)


def test_log_axis_detected():
    # 1, 10, 100, 1000 at equal pixel spacing -> log axis
    m = run_stub(fit_axis_map, ticks([(100, 1), (200, 10), (300, 100), (400, 1000)]),
                 scale_hint="unknown")
    assert m.kind == "log"
    np.testing.assert_allclose(m.to_data(np.array([250.0])), [10 ** 1.5], rtol=0.01)


def test_too_few_ticks_raises():
    with pytest.raises((ValueError, NotImplementedError)):
        fit_axis_map(ticks([(100, 200)]))
