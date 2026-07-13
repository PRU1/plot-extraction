# Implementation guide (the ★ stubs)

Order of attack: **calibration → tracing → signal → verify**. Each stage
unlocks the next in `p2s extract`, and `pytest -rs` is the scoreboard —
tests skip until your implementation exists, then score you against
exact synthetic ground truth.

---

## What to read first

From the paper (Jiang et al., arXiv:2107.02827):

- **Sec. 3.2 (plot data extraction)** — read this *well*. It's the
  intellectual core and maps directly onto `tracing.py`: the optical-flow
  analogy (x as time, curves as trajectories), the color-constancy and
  smoothness terms, and how they resolve crossings. Work through why
  embedding-based instance segmentation fails for plot lines (Sec. 2.3
  gives the argument) — it justifies the whole deterministic approach.
- **Sec. 3.1 (axis alignment)** — skim for the *problem statement*, not
  the solution. Their detector+refinement machinery is replaced here by
  projection profiles (`axes.py`, already working), but their Fig. 2
  makes the key point you'll carry into `calibration.py`: axis position
  error is a **systematic** error on every extracted point.
- **Sec. 4 (experiments)** — skim to see failure modes they report;
  those are your test cases.
- Skip Sec. 2.1/2.2 (detector and OCR surveys) — that machinery is
  Claude's job in this rewrite.

Background that pays off:

- Horn–Schunck / Lucas–Kanade optical flow (any vision-course notes):
  brightness constancy + smoothness regularization. You only need the
  *structure* of the objective, not the dense-flow solvers.
- Linear assignment problem / Hungarian algorithm — what
  `scipy.optimize.linear_sum_assignment` solves and its cost-matrix
  formulation.
- Eilers & Boelens (2005), *Baseline correction with asymmetric least
  squares smoothing* — short, readable, directly implementable.
- Savitzky & Golay (1964) — or any derivation showing the sliding LSQ
  polynomial fit is a convolution.

---

## 1. `calibration.py` — pixel → data map

Smallest stub; do it first to get a feel for the codebase.

1. Start with the clean linear fit: set up `A = [[px_i, 1]]`,
   solve the normal equations `(AᵀA)β = Aᵀv` (or `np.linalg.lstsq`).
   Two tests pass immediately.
2. Log detection: fit both `v ~ px` and `log10(v) ~ px` (latter only if
   all v > 0), compare residuals *in a comparable space*. Think about
   what noise model each regression assumes — that's the interesting
   subtlety. Cross-check the Claude `scale` hint; log disagreements to
   `Calibration.report`.
3. Robustness: exploit that ticks are equally spaced in value —
   `diff(v_i)` should be constant on a linear axis. A single OCR
   misread shows up as two anomalous diffs. Either RANSAC over tick
   pairs or iterative rejection on `|r_i| > k·MAD` works; implement one,
   understand why it breaks with < 4 ticks.
4. The y-flip (image rows increase downward) lives here and only here.
   Get the normalized-y fallback right: `x_axis_row → 0`, plot top → 1.

Checkpoint: `pytest tests/test_calibration.py` all green, and
`p2s extract` now stops at tracing instead.

## 2. `tracing.py` — the paper's core, and the hardest

Read paper Sec. 3.2 before writing code. Suggested path:

1. `extract_column_blobs`: per column, group foreground pixels into
   vertical runs → (centroid, top, bottom). Pure numpy; test it
   yourself on `synthetic.make_crossing_mask()` before going further.
2. Single curve, single color: greedy nearest-neighbor linking column
   to column already works. Get end-to-end plumbing running with this.
3. Add the state (y, y′) and the cost function from the module
   docstring; replace greedy with `linear_sum_assignment` per column
   pair. Tune σ_y, λ₁ on the crossing test — `test_two_same_color_
   curves_crossing` fails under greedy linking and passes once the
   smoothness term carries identity through the crossing. That single
   test failing→passing IS the paper's contribution, reproduced.
4. Then the long tail: gaps (dashed lines), track birth/death, steep
   segments where one column holds many pixels (blob centroid ≠ curve —
   match against blob intervals instead). Use `n_expected_curves` priors
   to seed track count.

Checkpoint: both tracing tests green; `p2s extract` stops at signal.

## 3. `signal.py` — four independent functions

Any order; each has one test.

- `resample_uniform`: pick an interpolant and justify it (PCHIP is a
  good default for spectra: monotone, no overshoot at peaks). Mark
  interpolated gap regions in provenance.
- `smooth`: derive the SG coefficients once from the Vandermonde
  pseudoinverse — then it's fine to call `scipy.signal.savgol_filter`.
  The derivation is the point; know why peak heights survive (moment
  preservation up to the polynomial order).
- `baseline`: implement ALS from the Eilers paper with
  `scipy.sparse` — banded system `(W + λD₂ᵀD₂)z = Wy`, iterate weights
  ~10×. Play with λ (stiffness) and p (asymmetry) on the test case
  until their roles are intuitive.
- `find_peaks_physical`: thin wrapper over `scipy.signal.find_peaks`,
  but with thresholds converted to physical units through the grid
  spacing. Optional: local Lorentzian fit for sub-pixel peak centers.

Checkpoint: full `p2s extract` runs; `reconstruction.png` should look
like the source figure. That visual comparison is the seed of stage 4.

## 4. `verify.py` — last, once the open loop works

No tests scaffolded — design it yourself (see the module docstring for
the render→compare→adjust structure). Before involving Claude, add the
cheap quantitative channel: ink-mask IoU between source and
reconstruction after aligning plot boxes. It catches gross failures for
free and gives the loop a convergence metric.

---

## Working habits that will save you pain

- One coordinate convention (image: origin top-left, row down)
  everywhere except inside calibration. Every plot-digitizer bug story
  starts with a y-flip in the wrong place.
- Keep stages pure functions of (input, params) — the verify loop
  re-runs stages with adjusted params and depends on this.
- When a test fails, render the failure: `cv2.imwrite` the mask with
  your trace drawn on it. Ten minutes of plotting beats an hour of
  staring at assertion diffs.
- Tunables go in `config.py` or function kwargs, never inline constants
  — the verify loop will need to reach them.
