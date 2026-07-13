"""Pixel -> data coordinate calibration.               *** YOURS TO IMPLEMENT ***

Inputs : AxesGeometry with x_ticks / y_ticks = [(px_i, v_i)] pairs, some
         v_i possibly None or misread (OCR is fallible), scale hints
         ("linear"/"log"/"unknown") from Claude that should be VERIFIED,
         not trusted.
Output : a Calibration that maps (col, row) pixel points to (x, y) data.

Math to work out (roughly in order of difficulty):

1. LINEAR AXIS, CLEAN TICKS.
   Model v = a*px + b. With ticks {(px_i, v_i)}, least squares:
       minimize  sum_i (a*px_i + b - v_i)^2
   Normal equations give a, b in closed form. Two ticks determine the
   map; more ticks overdetermine it -- that redundancy is your error
   detector, don't throw it away by using only the endpoints.

2. LOG AXIS DETECTION + FIT.
   On a log axis, equal pixel steps multiply the value: v = 10^(a*px+b).
   Detection: compare goodness-of-fit of regressing v on px vs.
   log10(v) on px (require all v_i > 0 for the latter). Prefer a proper
   criterion over raw R^2 -- both models have 2 parameters here, but
   think about what residual distribution you're assuming in each space.
   Cross-check against the Claude 'scale' hint; disagreement is worth a
   warning in the report.

3. ROBUSTNESS (the actually interesting part).
   Failure modes to handle:
     - one OCR value wrong (e.g. '8' read as '3'): a single gross outlier.
       Options: RANSAC over tick pairs; or iterate LSQ with residual-based
       rejection (|r_i| > k*MAD). With >= 4 ticks you can always identify
       one bad value, since ticks are *equally spaced in data value* on a
       linear axis -- exploit that structure: diff(v_i) should be constant.
     - counts mismatch (n pixels != n values): match_ticks() zipped
       best-effort. The consistent-spacing structure again identifies
       the correct alignment: try each alignment offset, keep the one
       with the lowest robust fit residual.
     - value=None entries: just exclude from the fit.
   Report per-axis: fitted params, residual RMS in data units, number of
   rejected ticks. Downstream (and the future verify loop) uses this.

4. Y-AXIS POLICY.
   Spectra usually have arbitrary intensity units. If y ticks are absent
   or unmatched, fall back to a normalized map: row -> [0, 1] over the
   plot bbox (top row -> 1, x_axis_row -> 0). Flag it in the report.

5. THE FLIP. Image rows increase downward; data y increases upward.
   Handle it HERE and nowhere else (see models.py docstring).

Suggested self-check: tests/test_calibration.py fabricates tick sets
(clean, noisy, one-outlier, log) with known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .models import AxesGeometry


@dataclass
class AxisMap:
    """One axis' pixel->data map: value = a*px + b (linear) or 10^(a*px+b) (log)."""

    kind: Literal["linear", "log", "normalized"]
    a: float
    b: float

    def to_data(self, px: np.ndarray) -> np.ndarray:
        lin = self.a * np.asarray(px, dtype=float) + self.b
        return 10.0 ** lin if self.kind == "log" else lin


@dataclass
class Calibration:
    x_map: AxisMap
    y_map: AxisMap
    report: dict = field(default_factory=dict)  # residuals, rejected ticks, warnings

    def px_to_data(self, points_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,2) array of (col, row) -> (x_data, y_data) arrays."""
        return self.x_map.to_data(points_px[:, 0]), self.y_map.to_data(points_px[:, 1])


def fit_axis_map(ticks: list, scale_hint: str = "unknown") -> AxisMap:
    """Fit one axis' map from [(px, value)] ticks. See module docstring.

    Must handle: value=None entries, a single gross OCR outlier,
    log-axis detection (verify the hint, don't trust it).
    Raise ValueError with a clear message if fewer than 2 usable ticks
    survive rejection.
    """
    raise NotImplementedError("fit_axis_map: your calibration math goes here")


def calibrate(geometry: AxesGeometry) -> Calibration:
    """Build the full Calibration from AxesGeometry.

    - x axis: fit_axis_map(geometry.x_ticks, geometry.x_scale)
    - y axis: fit_axis_map(...) if usable ticks exist, else the
      normalized fallback (docstring item 4) using plot_bbox/x_axis_row.
    - populate Calibration.report (fit residuals, rejections, warnings).
    """
    raise NotImplementedError("calibrate: your calibration math goes here")
