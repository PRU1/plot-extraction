"""Signal processing on digitized spectra.                    *** YOURS ***

Input : CurveTrace (pixel points) + Calibration, or a raw (x, y) pair.
Output: Spectrum objects on a clean, uniform grid, plus derived features.

Functions to implement, with the relevant theory to think through:

1. resample_uniform(x, y, n) -> (x_u, y_u)
   Traced points are one-per-pixel-column: nearly uniform in x for a
   linear axis, exponentially spaced for a log axis, and possibly with
   gaps (dashed lines, crossings). Choose and justify an interpolant:
   linear (safe, adds spurious curvature kinks), cubic spline (smooth,
   can ring near sharp peaks -- compare natural vs. not-a-knot), or
   monotone PCHIP (no overshoot; good default for spectra). Consider
   what happens at gaps: interpolating across a wide gap fabricates data
   -- carry a mask of interpolated regions in Spectrum.provenance.

2. smooth(y, ...) -> y_s
   Savitzky-Golay is the standard for spectra: fit a degree-p polynomial
   over a sliding window of m points by least squares, evaluate at the
   center. Because LSQ is linear in the data, the whole thing is a
   CONVOLUTION with fixed coefficients -- derive them once from the
   pseudoinverse of the Vandermonde matrix (or call scipy.signal.savgol_
   filter after you've derived why it works). Key property vs. moving
   average: preserves moments up to order p, so peak heights/widths
   survive. Choose window << peak FWHM.

3. baseline(y, ...) -> y_b
   Asymmetric least squares (Eilers & Boelens 2005). Objective:
       minimize_z  sum_i w_i (y_i - z_i)^2 + lam * sum_i (D2 z)_i^2
   where D2 is the second-difference operator and weights are
   asymmetric: w_i = p if y_i > z_i else (1-p), with p ~ 0.001-0.01.
   Peaks (y above baseline) get tiny weight, so z hugs the valleys.
   Iterate: solve the banded linear system (W + lam*D2^T D2) z = W y,
   update w, repeat ~10x. scipy.sparse makes this a few lines -- the
   interesting part is understanding lam (stiffness) vs. p (asymmetry).

4. find_peaks_physical(spectrum, ...) -> list[Peak]
   Wrap scipy.signal.find_peaks but express thresholds in physical
   units: prominence as a fraction of the y range, min distance in x
   units (cm^-1, eV) converted through the grid spacing. Optionally
   refine each peak by fitting a local Lorentzian/Gaussian -- sub-pixel
   peak positions are one of the main reasons to digitize at all.

Ordering note: baseline-correct BEFORE measuring peak prominence;
smooth AFTER resampling (uniform grid assumed by SG); and keep every
operation recorded in Spectrum.provenance for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import Calibration
from .models import CurveTrace, Spectrum


@dataclass
class Peak:
    x: float
    y: float
    prominence: float
    fwhm: float | None = None


def trace_to_spectrum(trace: CurveTrace, calibration: Calibration,
                      n_points: int = 2000) -> Spectrum:
    """CurveTrace -> calibrated, uniformly resampled Spectrum.

    Steps: px_to_data via calibration -> sort by x -> resample_uniform
    -> record provenance (trace color, calibration report, n_points).
    """
    raise NotImplementedError("trace_to_spectrum: yours")


def resample_uniform(x: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("resample_uniform: yours (see module docstring, item 1)")


def smooth(y: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    raise NotImplementedError("smooth: yours (item 2)")


def baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
             n_iter: int = 10) -> np.ndarray:
    raise NotImplementedError("baseline: yours (item 3)")


def find_peaks_physical(spectrum: Spectrum, min_prominence_frac: float = 0.05,
                        min_distance_x: float | None = None) -> list[Peak]:
    raise NotImplementedError("find_peaks_physical: yours (item 4)")
