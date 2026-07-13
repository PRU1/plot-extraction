"""Axis alignment: locate axis lines, tick marks, and read tick values.

This replaces the original paper's Stage 1 (anchor-free detector +
edge-based bbox refinement + CRAFT scene-text OCR) with a training-free
split:

  GEOMETRY (pixel-precise)  -> classical CV, here.
      Axis lines via maximum dark-run projection: the y-axis is the
      column in the left part of the image with the longest vertical
      run of dark pixels; likewise the x-axis among rows at the bottom.
      This is the degenerate-but-robust version of a Hough transform
      when lines are known to be axis-parallel. It plays the same role
      as the paper's "edge-based constraint": snap to the strongest
      straight edge, because axis position error is a systematic error
      on every extracted point.

      Tick marks via peak-finding on the dark-pixel density of a thin
      strip just OUTSIDE the plot box (ticks protrude outward or text
      sits there; either way the strip is periodic in tick spacing).

  SEMANTICS (values, labels, scale) -> Claude vision (claude.py).
      Claude reads the tick VALUES from a cropped label band. It never
      supplies pixel positions.

  FUSION -> match_ticks(): pairs CV tick pixels with OCR values. The
      naive zip is done here; making the pairing robust to OCR misses /
      extra detections belongs with the calibration math (yours), where
      regression residuals reveal bad pairs.
"""

from __future__ import annotations

import io

import cv2
import numpy as np

from .claude import ClaudeVision, OfflineError
from .config import Config
from .models import AxesGeometry, BBox, TickMark


# ----------------------------------------------------------------------
# Binarization
# ----------------------------------------------------------------------

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def dark_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean mask of 'ink' pixels via Otsu threshold (dark on light)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary > 0


def _max_run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Longest consecutive run of True along `axis`, per line.

    E.g. axis=0: for every column, the longest vertical run of dark
    pixels. A solid axis line has a run comparable to the image height;
    text and curves do not.
    """
    n = mask.shape[axis]
    x = np.moveaxis(mask, axis, 0).astype(np.int32)  # (n, other)
    best = np.zeros(x.shape[1], dtype=np.int32)
    cur = np.zeros(x.shape[1], dtype=np.int32)
    for i in range(n):
        cur = (cur + 1) * x[i]
        best = np.maximum(best, cur)
    return best


# ----------------------------------------------------------------------
# Axis line detection
# ----------------------------------------------------------------------

def detect_axis_lines(img_bgr: np.ndarray, cfg: Config | None = None) -> tuple[int, int, BBox]:
    """Return (x_axis_row, y_axis_col, plot_bbox).

    plot_bbox is the interior region: right of the y-axis, above the
    x-axis, bounded on the far sides by the end of the axis lines (or a
    detected top/right frame if the plot is boxed).
    """
    cfg = cfg or Config()
    mask = dark_mask(to_gray(img_bgr))
    h, w = mask.shape

    # y-axis: column with the longest vertical dark run, searched in the left region
    col_runs = _max_run_lengths(mask, axis=0)
    left = int(w * cfg.axis_search_frac)
    y_candidates = np.where(col_runs[:left] >= cfg.min_axis_run_frac * h)[0]
    y_axis_col = int(y_candidates[-1]) if len(y_candidates) else int(np.argmax(col_runs[:left]))
    # `[-1]`: if a boxed frame gives two tall columns in the left region,
    # the *rightmost* one adjacent to the labels is the y-axis... except a
    # frame's left edge IS the y-axis; taking the last candidate handles
    # double-line artifacts from anti-aliasing.

    # x-axis: row with the longest horizontal dark run, searched in the bottom region
    row_runs = _max_run_lengths(mask, axis=1)
    bottom_start = int(h * (1 - cfg.axis_search_frac))
    x_candidates = bottom_start + np.where(row_runs[bottom_start:] >= cfg.min_axis_run_frac * w)[0]
    x_axis_row = int(x_candidates[0]) if len(x_candidates) else int(bottom_start + np.argmax(row_runs[bottom_start:]))
    # `[0]`: topmost long row in the bottom region -- below it there is
    # only tick text (and possibly a frame bottom edge duplicate).

    # top/right bounds: matching frame lines if boxed, else image edge
    right_candidates = np.where(col_runs[left:] >= cfg.min_axis_run_frac * h)[0]
    right = int(left + right_candidates[-1]) if len(right_candidates) else w - 1
    top_candidates = np.where(row_runs[: h - bottom_start] >= cfg.min_axis_run_frac * w)[0]
    top = int(top_candidates[0]) if len(top_candidates) else 0

    plot_bbox = BBox(y_axis_col + 1, top, right, x_axis_row - 1)
    return x_axis_row, y_axis_col, plot_bbox


# ----------------------------------------------------------------------
# Tick localization
# ----------------------------------------------------------------------

def locate_ticks(img_bgr: np.ndarray, x_axis_row: int, y_axis_col: int,
                 plot_bbox: BBox, axis: str = "x", cfg: Config | None = None) -> list[float]:
    """Pixel positions of tick marks along one axis.

    Method: take a thin strip just outside the plot box (ticks protrude
    outward in most journal styles; inward ticks still leave their label
    text in the strip), sum ink density across the strip thickness, and
    find peaks separated by at least tick_min_separation_px.
    """
    from scipy.signal import find_peaks

    cfg = cfg or Config()
    mask = dark_mask(to_gray(img_bgr))
    h, w = mask.shape

    pad = 5  # corner ticks sit exactly on the plot-box boundary
    if axis == "x":
        lo = max(int(plot_bbox.x0) - pad, 0)
        strip = mask[x_axis_row + 2 : min(x_axis_row + 2 + cfg.tick_strip_px, h),
                     lo : min(int(plot_bbox.x1) + pad, w)]
        density = strip.sum(axis=0).astype(float)
        offset = lo
    else:
        lo = max(int(plot_bbox.y0) - pad, 0)
        strip = mask[lo : min(int(plot_bbox.y1) + pad, h),
                     max(y_axis_col - cfg.tick_strip_px - 1, 0) : y_axis_col - 1]
        density = strip.sum(axis=1).astype(float)
        offset = lo

    if density.max() <= 0:
        return []
    peaks, _ = find_peaks(density, height=0.5 * density.max(),
                          distance=cfg.tick_min_separation_px)
    # Sub-pixel refinement: centroid of the density in a +/-2 px window.
    refined = []
    for p in peaks:
        lo, hi = max(p - 2, 0), min(p + 3, len(density))
        window = density[lo:hi]
        refined.append(offset + lo + float((window * np.arange(len(window))).sum() / window.sum()))
    return refined


# ----------------------------------------------------------------------
# OCR fusion
# ----------------------------------------------------------------------

def match_ticks(tick_pxs: list[float], values: list[float]) -> list[TickMark]:
    """Pair CV tick pixels with OCR values (both in increasing-px order).

    Naive policy: zip when counts match; otherwise attach values=None and
    let the calibration stage decide (a robust fit over the subset of
    matched ticks, with residual-based rejection, is the right tool --
    that math is yours, see calibration.py).
    """
    if len(tick_pxs) == len(values):
        return [TickMark(px=p, value=v) for p, v in zip(tick_pxs, values)]
    ticks = [TickMark(px=p) for p in tick_pxs]
    for i, v in enumerate(values[: len(ticks)]):
        ticks[i].value = v  # best-effort; calibration must validate
    return ticks


def _crop_png(img_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> bytes:
    x0, y0, x1, y1 = bbox
    crop = img_bgr[max(y0, 0) : y1, max(x0, 0) : x1]
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def read_axes(img_bgr: np.ndarray, cfg: Config | None = None,
              vision: ClaudeVision | None = None) -> AxesGeometry:
    """Full axis-alignment stage: geometry (CV) + semantics (Claude)."""
    cfg = cfg or Config.from_env()
    vision = vision or ClaudeVision(cfg)
    h, w = img_bgr.shape[:2]

    x_axis_row, y_axis_col, plot_bbox = detect_axis_lines(img_bgr, cfg)
    x_tick_px = locate_ticks(img_bgr, x_axis_row, y_axis_col, plot_bbox, "x", cfg)
    y_tick_px = locate_ticks(img_bgr, x_axis_row, y_axis_col, plot_bbox, "y", cfg)

    geometry = AxesGeometry(
        x_axis_row=x_axis_row, y_axis_col=y_axis_col, plot_bbox=plot_bbox,
        x_ticks=[TickMark(px=p) for p in x_tick_px],
        y_ticks=[TickMark(px=p) for p in y_tick_px],
    )

    # Semantic layer: crop the label bands and let Claude read them.
    try:
        x_band = _crop_png(img_bgr, (int(plot_bbox.x0) - 20, x_axis_row, w, h))
        x_info = vision.read_ticks(x_band, axis="x")
        geometry.x_ticks = match_ticks(x_tick_px, [float(v) for v in x_info.get("tick_values", [])])
        geometry.x_label = x_info.get("axis_label")
        geometry.x_scale = x_info.get("scale", "unknown")

        y_band = _crop_png(img_bgr, (0, int(plot_bbox.y0) - 10, y_axis_col, x_axis_row + 10))
        y_info = vision.read_ticks(y_band, axis="y")
        geometry.y_ticks = match_ticks(y_tick_px, [float(v) for v in y_info.get("tick_values", [])])
        geometry.y_label = y_info.get("axis_label")
        geometry.y_scale = y_info.get("scale", "unknown")
    except OfflineError:
        pass  # geometry-only result; calibration will complain if it needs values

    return geometry
