"""Closed-loop verification (Option-3 bolt-on).          *** YOURS, LATER ***

Do not build this until calibration + tracing + signal work open-loop.

Idea: the pipeline above is open-loop -- errors in any stage propagate
silently. Close the loop:

  1. RENDER: re-plot the extracted Spectrum objects with matplotlib,
     matched to the source figure (same axis ranges via the calibration,
     same colors, same size). export.replot() already does most of this.
  2. COMPARE: send (original figure, reconstruction) to Claude vision:
     "Same curves? Where do they diverge? Missing/extra curves?
      Axis ranges consistent?" -> structured discrepancy report.
     Optionally add a quantitative channel: IoU of the ink masks after
     aligning plot bboxes -- cheap, catches gross errors without a
     vision call.
  3. ADJUST: map each discrepancy type to a parameter change and re-run
     the offending stage:
       "missing curve"        -> segmentation k / min_curve_pixels
       "curves swapped after
        crossing at x~X"      -> tracing lambda_1 (smoothness weight)
       "x range off by ~10x"  -> calibration log/linear decision
       "curve truncated"      -> tracing gap penalty
  4. TERMINATE on convergence (report clean / IoU plateau) or max_iters.

Design constraints:
  - every iteration must be logged (params in, report out) -- this loop
    is an experiment; treat it like one.
  - stages are already re-entrant pure functions of (input, params),
    which is what makes this loop possible. Keep them that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VerificationResult:
    converged: bool
    iterations: int
    final_report: dict
    history: list[dict] = field(default_factory=list)


class VerificationLoop:
    def __init__(self, config=None, vision=None, max_iters: int = 3):
        self.config = config
        self.vision = vision
        self.max_iters = max_iters

    def run(self, figure_path: Path, workdir: Path) -> VerificationResult:
        raise NotImplementedError("VerificationLoop.run: build after the open-loop pipeline works")
