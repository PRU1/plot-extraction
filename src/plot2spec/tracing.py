"""Curve tracing: group curve pixels into individual curves.   *** YOURS ***

This is the paper's core deterministic contribution (Sec. 3.2): instance
assignment via optical-flow-style constraints, no learning.

Inputs : CurveMask list from segmentation (one mask per curve COLOR; a
         mask may contain SEVERAL curves, e.g. offset stacks), plot bbox,
         and per-mask n_expected_curves priors from Claude (may be None).
Output : CurveTrace list -- per curve, an (N,2) array of (col, row)
         pixel points, one point (or none) per column.

The idea, in the paper's framing
--------------------------------
Treat the column index x as TIME and each curve as a particle trajectory
y(x). Two physical constraints identify trajectories:

  (i)  color/brightness constancy -- a curve keeps its color along its
       length (the optical-flow brightness-constancy term);
  (ii) smoothness -- dy/dx changes slowly (spectra are continuous;
       even sharp Raman peaks are many pixels wide at plot resolution).

Sketch of one clean formulation
-------------------------------
1. COLUMN BLOBS. For each column x in the mask, group foreground pixels
   into connected vertical runs; each run's centroid y (and its vertical
   extent) is one observation. A column may have fewer observations than
   curves (crossings, occlusion) or more (noise, markers).

2. STATE + COST. Track state s = (y, y') per curve. Predict the next
   column: y_pred = y + y' * dx. Cost of assigning observation q in
   column x+dx to track p:

       J(p, q) = (y_q - y_pred_p)^2 / sigma_y^2
               + lambda_1 * (y'_new - y'_p)^2
               + lambda_2 * ||c_q - c_p||^2        (only if colors shared)

   The first two terms are the discrete smoothness/flow terms; the third
   is brightness constancy. (This is a Kalman-filter-flavored rewriting
   of the paper's optical flow objective -- equivalent structure, easier
   to implement column-by-column.)

3. ASSIGNMENT. Per column pair, solve the rectangular assignment problem
   minimizing total J -- scipy.optimize.linear_sum_assignment. Allow
   'no match' (track skips a column) with a gap penalty; cap consecutive
   gaps before terminating a track.

4. CROSSINGS. At an intersection two tracks compete for one blob or for
   merged blobs. The smoothness term resolves it: the assignment that
   preserves each track's slope wins. This is precisely why straight-
   through crossings work and why identical curves that touch
   tangentially are fundamentally ambiguous (know this failure mode).

5. TRACK MANAGEMENT. Initialize tracks in the first few columns (use
   n_expected_curves if given: seed k-means on blob y-positions, or take
   the n most persistent blobs). Merge/prune fragments at the end;
   enforce one point per column per track (or none).

Edge cases worth designing for from the start:
  - vertical extent: a near-vertical segment (steep peak edge) puts many
    pixels in one column -- the blob centroid is NOT on the curve there.
    Consider matching against blob (top, bottom) intervals, or handling
    steep segments by swapping the roles of x and y locally.
  - same-color offset stacks: no color term available; smoothness and
    the n_expected prior do all the work.
  - dashed/dotted lines: gaps are structural, not noise -- gap penalty
    must tolerate the dash period.
  - markers (circles/squares on the line): fat blobs; centroid ok.

Tuning: sigma_y, lambda_1, lambda_2, gap penalty. Expose them as
function parameters with defaults; tests/test_tracing.py generates
crossing ground-truth curves to score against.
"""

from __future__ import annotations

import numpy as np

from .models import BBox, CurveMask, CurveTrace


def extract_column_blobs(mask: np.ndarray) -> list[list[tuple[float, float, float]]]:
    """Per column: list of (y_centroid, y_top, y_bottom) for each vertical
    run of foreground pixels.

    Provided as a suggested first sub-step; feel free to replace.
    """
    raise NotImplementedError("extract_column_blobs: yours")


def trace_curves(masks: list[CurveMask], plot_bbox: BBox) -> list[CurveTrace]:
    """Group curve pixels into individual CurveTraces. See module docstring."""
    raise NotImplementedError("trace_curves: your trajectory-linking math goes here")
