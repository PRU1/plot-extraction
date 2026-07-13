"""Tests for YOUR signal processing. Skip until implemented."""

import numpy as np

from plot2spec.models import Spectrum
from plot2spec.signal import baseline, find_peaks_physical, resample_uniform, smooth

from conftest import run_stub
from synthetic import lorentzian


def test_resample_uniform_recovers_function():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 10, 300))          # non-uniform sampling
    y = np.sin(x)
    xu, yu = run_stub(resample_uniform, x, y, 500)
    assert len(xu) == len(yu) == 500
    assert np.allclose(np.diff(xu), np.diff(xu)[0])          # uniform grid
    interior = (xu > 0.5) & (xu < 9.5)
    assert np.max(np.abs(yu[interior] - np.sin(xu[interior]))) < 0.01


def test_smooth_reduces_noise_keeps_peak():
    x = np.linspace(0, 100, 1000)
    clean = lorentzian(x, 50, 5)
    noisy = clean + np.random.default_rng(1).normal(0, 0.05, x.shape)
    ys = run_stub(smooth, noisy, window=11, polyorder=3)
    assert np.std(ys - clean) < 0.5 * np.std(noisy - clean)   # noise reduced
    assert abs(ys.max() - clean.max()) < 0.05                 # peak height preserved


def test_baseline_hugs_valleys():
    x = np.linspace(0, 100, 1000)
    true_base = 0.5 + 0.01 * x
    y = true_base + 2 * lorentzian(x, 30, 3) + 2 * lorentzian(x, 70, 3)
    zb = run_stub(baseline, y)
    off_peak = (np.abs(x - 30) > 15) & (np.abs(x - 70) > 15)
    assert np.max(np.abs(zb[off_peak] - true_base[off_peak])) < 0.1


def test_find_peaks_physical():
    x = np.linspace(100, 1000, 2000)
    y = lorentzian(x, 300, 20) + 0.6 * lorentzian(x, 700, 15)
    s = Spectrum(x=x, y=y)
    peaks = run_stub(find_peaks_physical, s, min_prominence_frac=0.1)
    found = sorted(p.x for p in peaks)
    assert len(found) == 2
    assert abs(found[0] - 300) < 5 and abs(found[1] - 700) < 5
