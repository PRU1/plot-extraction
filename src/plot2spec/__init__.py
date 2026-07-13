"""plot2spec-v2: paper PDF -> figure crops -> spectra data.

Architecture (see README.md for the full picture):

    Stage A (ingest)      pdf -> FigureCandidate[]      ingest.py    [working]
    Stage B (extract)
        axes/ticks        image -> AxesGeometry         axes.py      [working]
        calibration       AxesGeometry -> Calibration   calibration.py  [YOURS]
        segmentation      image -> CurveMask[]          segmentation.py [working]
        tracing           CurveMask[] -> CurveTrace[]   tracing.py      [YOURS]
        signal            CurveTrace -> Spectrum        signal.py       [YOURS]
    Verify loop           closed-loop check             verify.py       [YOURS, later]

Modules marked [YOURS] are intentionally left as documented stubs.
"""

__version__ = "0.1.0"
