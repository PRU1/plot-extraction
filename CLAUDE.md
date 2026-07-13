# CLAUDE.md — agent instructions for this repo

## What this is
plot2spec-v2: PDF → figure crops → digitized spectra. Claude vision for
semantics, classical CV for geometry, deterministic math in between.
Read README.md for the architecture.

## Hard rule: the ★ stubs are Pranav's
`calibration.py`, `tracing.py`, `signal.py`, `verify.py` are
intentionally unimplemented — Pranav is implementing the math himself
to learn it. **Do not implement, partially implement, or "helpfully
fix" these modules unless he explicitly asks.** Reviewing his
implementations, discussing the math, and improving their docstrings
are all fine.

## Conventions
- Pixel coordinates: image convention everywhere (origin top-left, row
  increases downward). The y-flip to data coordinates happens ONLY in
  `calibration.py`. See `models.py` docstring.
- Pipeline stages are pure-ish functions of (input, Config); keep them
  re-entrant — the future verify loop depends on it.
- All Claude calls go through `claude.ClaudeVision.ask_json` (cached,
  strict JSON). Never call the anthropic SDK directly from stages.
- Claude supplies semantics only (values/labels/classes/counts), never
  pixel positions — patch-granularity localization is not trustworthy.

## Commands
```bash
pip install -e ".[dev]"    # setup
pytest                     # offline; stub tests skip via conftest.run_stub
pytest -rs                 # show which stubs are pending
p2s extract <img> -o out/  # runs until the first unimplemented stub
```

## When editing
- New tunables go in `config.py`, not as magic numbers.
- New Claude prompts go in `claude.py` as module-level constants.
- Tests must stay offline (no API key required): use `vision=None` or
  rely on OfflineError degradation.
