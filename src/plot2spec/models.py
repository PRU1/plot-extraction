"""Typed data models passed between pipeline stages.

Convention: all pixel coordinates use image convention — origin at the
TOP-LEFT, x increasing rightward (columns), y increasing DOWNWARD (rows).
The flip to data convention (y up) happens exactly once, inside the
calibration mapping. Keeping one convention everywhere else avoids the
single most common plot-digitizer bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np


# ----------------------------------------------------------------------
# Geometry primitives
# ----------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned box in pixel coordinates (x0, y0) top-left, (x1, y1) bottom-right."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def padded(self, pad: float, bounds: "BBox | None" = None) -> "BBox":
        b = BBox(self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad)
        if bounds is not None:
            b = BBox(
                max(b.x0, bounds.x0), max(b.y0, bounds.y0),
                min(b.x1, bounds.x1), min(b.y1, bounds.y1),
            )
        return b

    def iou(self, other: "BBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x0), int(self.y0), int(self.x1), int(self.y1)


# ----------------------------------------------------------------------
# Stage A: ingest
# ----------------------------------------------------------------------

@dataclass
class Panel:
    """One sub-plot of a multi-panel figure, e.g. panel '(b)'."""

    image_path: Path
    label: str | None = None            # "(a)", "(b)", ... if identified
    is_spectrum_plot: bool | None = None


@dataclass
class FigureCandidate:
    """A figure pulled out of a PDF, before/after classification."""

    image_path: Path
    source: Literal["embedded", "render"]
    page_index: int                      # 0-based
    bbox_on_page: BBox | None = None     # in rendered-page pixel coords (render source)
    caption: str | None = None
    is_spectrum_plot: bool | None = None
    plot_kind: str | None = None         # e.g. "raman", "xanes", "absorption", "other"
    panels: list[Panel] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["image_path"] = str(self.image_path)
        for p in d["panels"]:
            p["image_path"] = str(p["image_path"])
        return d


# ----------------------------------------------------------------------
# Stage B: axes & calibration
# ----------------------------------------------------------------------

@dataclass
class TickMark:
    """A tick on one axis.

    px    : pixel coordinate ALONG the axis (column index for x-axis
            ticks, row index for y-axis ticks).
    value : numeric value read by OCR, or None if unmatched. Robustly
            handling None / misread values is the calibration stub's job.
    """

    px: float
    value: float | None = None


@dataclass
class AxesGeometry:
    """Output of the axis-alignment stage (paper Stage 1 equivalent)."""

    x_axis_row: int                      # pixel row of the x-axis line
    y_axis_col: int                      # pixel column of the y-axis line
    plot_bbox: BBox                      # interior plot region
    x_ticks: list[TickMark] = field(default_factory=list)
    y_ticks: list[TickMark] = field(default_factory=list)
    x_label: str | None = None           # e.g. "Raman shift (cm$^{-1}$)"
    y_label: str | None = None           # usually "Intensity (a.u.)"
    x_scale: Literal["linear", "log", "unknown"] = "unknown"
    y_scale: Literal["linear", "log", "unknown"] = "unknown"


# ----------------------------------------------------------------------
# Stage B: segmentation & tracing
# ----------------------------------------------------------------------

@dataclass
class CurveMask:
    """Binary mask of pixels believed to belong to curves of one color.

    NOTE: several curves may share one color (offset-stacked spectra are
    the classic case) — separating them is the tracing stage's job.
    """

    mask: np.ndarray                     # bool array, full-image shape
    color_rgb: tuple[int, int, int]      # representative color (cluster center)
    n_expected_curves: int | None = None # prior from Claude, if available


@dataclass
class CurveTrace:
    """One traced curve, still in pixel coordinates."""

    points_px: np.ndarray                # (N, 2) float array of (col, row); sorted by col
    color_rgb: tuple[int, int, int]
    label: str | None = None             # legend entry, if matched


# ----------------------------------------------------------------------
# Final output
# ----------------------------------------------------------------------

@dataclass
class Spectrum:
    """A digitized spectrum in data coordinates."""

    x: np.ndarray
    y: np.ndarray
    x_label: str | None = None
    y_label: str | None = None
    curve_label: str | None = None
    provenance: dict = field(default_factory=dict)  # source pdf, page, figure, params

    def to_json_dict(self) -> dict:
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "x_label": self.x_label,
            "y_label": self.y_label,
            "curve_label": self.curve_label,
            "provenance": self.provenance,
        }
