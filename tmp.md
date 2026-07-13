# plot2spec-v2

Paper PDF → figure crops → digitized spectra. A rewrite of the
[Plot2Spectra](https://github.com/MaterialEyes/Plot2Spec) idea
([Jiang et al., arXiv:2107.02827](https://arxiv.org/abs/2107.02827))
that replaces the trained perception networks (anchor-free detector,
CRAFT OCR, segmentation CNN) with a **Claude vision API layer for
semantics** and **classical CV for geometry** — no model weights, no
GPU, no 2021-pinned dependency stack — and adds the document layer the
original lacked (PDF ingestion, figure localization, captions, panels).

## Pipeline

```
                     ┌─────────────────────  Stage A: ingest.py  ──────────────────────┐
 paper.pdf ──────►   embedded XObject extraction (lossless, pymupdf)                    │
                     + page render → Claude vision figure bboxes (vector figs)          │
                     + captions from PDF text layer (Claude fallback)                   │
                     + Claude classification (spectrum? kind?) & panel splitting        │
                     └──────────────────────────┬───────────────────────────────────────┘
                                                │  FigureCandidate[] + figures.json
                     ┌─────────────────────  Stage B: extraction  ──────────────────────┐
 plot image ──────►  axes.py          axis lines + tick pixels (projection profiles)    │ working
                     claude.py        tick VALUES, labels, scale (vision OCR)           │ working
                     calibration.py   pixel→data map: LSQ, log detect, robust rejection │ ★ yours
                     segmentation.py  curve pixels by color (Lab k-means + priors)      │ working
                     tracing.py       pixels → individual curves (trajectory linking)   │ ★ yours
                     signal.py        resample, smooth (SG), baseline (ALS), peaks      │ ★ yours
                     export.py        CSV / JSON / re-plot                              │ working
                     └──────────────────────────┬───────────────────────────────────────┘
                                                │  Spectrum[] + reconstruction.png
                     verify.py        closed-loop render→compare→adjust                   ★ yours, later
```

Design principle: **Claude does semantics, classical CV does geometry,
your math fuses them.** Vision transformers read text and classify
superbly but localize only to patch granularity; axis lines and tick
marks need pixel precision, which projection profiles deliver for free.
The redundancy between the two channels (ticks are equally spaced in
value; regression residuals expose OCR misreads) is the built-in error
detector — exploit it in `calibration.py`.

## What you implement (★)

| Module | The math | Tests waiting for you |
|---|---|---|
| `calibration.py` | LSQ tick regression, log-axis detection, robust outlier rejection, y-flip | `tests/test_calibration.py` |
| `tracing.py` | column-blob trajectory linking: color constancy + slope smoothness, assignment problem, crossings | `tests/test_tracing.py` |
| `signal.py` | resampling/interpolation choice, Savitzky–Golay, ALS baseline, peak refinement | `tests/test_signal.py` |
| `verify.py` | closed-loop verification (build last) | — |

Each stub's module docstring contains the derivation road map and the
edge cases worth designing for. Tests **skip** while a stub raises
`NotImplementedError` and activate automatically once you implement it:

```
pytest              # today: axes+segmentation tests pass, your tests skip
pytest -rs          # see exactly which stubs the skips point at
```

Tests score against exact ground truth: `tests/synthetic.py` renders
matplotlib spectra and keeps every pixel coordinate via `ax.transData`
— no hand labeling, no API calls, fully offline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add your ANTHROPIC_API_KEY
```

## Usage

```bash
p2s ingest paper.pdf -o out/figures      # Stage A → crops + figures.json
p2s extract out/figures/p003_render_0_panel1.png -o out/extract
p2s run paper.pdf -o out                 # end to end
p2s replot-cmd out/extract/spectra.json  # visual sanity check
```

`p2s extract` runs as far as the implemented stages allow, saves all
intermediates (`axes.json`, `mask_*.png`), and tells you which stub it
stopped at — so the plumbing is exercisable from day one.

## Determinism & caching

Every Claude call is cached in `.cache/claude/` keyed by
sha256(model | prompt | image bytes). API sampling is not bit-exact
even at temperature 0; the cache is what makes reruns reproducible
(and free). Delete the cache to force fresh perception. With no API
key, the pipeline runs in offline mode: geometry-only axes, un-primed
segmentation — enough for all tests.

## References

- Jiang, Schwenker, Spreadbury, Li, Chan, Cossairt, *Plot2Spectra: an
  Automatic Spectra Extraction Tool*, arXiv:2107.02827 (2021).
- Eilers & Boelens, *Baseline correction with asymmetric least squares
  smoothing* (2005) — for `signal.baseline`.
- Savitzky & Golay, *Smoothing and differentiation of data by
  simplified least squares procedures*, Anal. Chem. (1964).
