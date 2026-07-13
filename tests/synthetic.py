"""Synthetic ground-truth figures for testing.

The trick: matplotlib KNOWS the exact pixel position of everything it
draws (ax.transData), so we can render a spectrum figure and keep the
pixel-level ground truth -- axis line positions, tick pixel coords, and
every curve's (col, row) trajectory. Extraction code is then scored
against exact answers, no hand labeling.

Coordinate note: matplotlib display coords have origin at BOTTOM-left;
image arrays have row 0 at the TOP. Everything returned here is already
converted to image convention (row = H - y_display).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SyntheticFigure:
    img_bgr: np.ndarray                 # (H, W, 3) uint8, as cv2.imread would give
    x_axis_row: float                   # image-convention row of the bottom spine
    y_axis_col: float                   # column of the left spine
    xtick_px: list[float]               # tick columns, left to right
    xtick_values: list[float]
    curves_px: list[np.ndarray] = field(default_factory=list)  # per curve (N,2) (col,row)
    curves_data: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    colors_rgb: list[tuple[int, int, int]] = field(default_factory=list)
    xlim: tuple[float, float] = (0.0, 1.0)


_COLORS = [(214, 39, 40), (44, 160, 44), (31, 119, 180)]  # tab:red/green/blue RGB


def lorentzian(x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
    return 1.0 / (1.0 + ((x - x0) / gamma) ** 2)


def make_spectrum_figure(
    n_curves: int = 3,
    seed: int = 0,
    dpi: int = 100,
    figsize: tuple[float, float] = (6.0, 4.0),
    xlim: tuple[float, float] = (100.0, 1000.0),
    offset_stack: bool = True,
    linewidth: float = 2.0,
) -> SyntheticFigure:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x = np.linspace(xlim[0], xlim[1], 800)
    curves_data = []
    for i in range(n_curves):
        centers = rng.uniform(xlim[0] + 100, xlim[1] - 100, size=3)
        widths = rng.uniform(15, 40, size=3)
        y = sum(lorentzian(x, c, g) for c, g in zip(centers, widths))
        if offset_stack:
            y = y + 1.6 * i
        color = tuple(c / 255 for c in _COLORS[i % len(_COLORS)])
        ax.plot(x, y, color=color, lw=linewidth)
        curves_data.append((x, y))

    ax.set_xlim(xlim)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    fig.canvas.draw()

    W, H = fig.canvas.get_width_height()
    bbox = ax.get_window_extent()
    ylim = ax.get_ylim()

    xticks = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
    xtick_px = [float(ax.transData.transform((t, ylim[0]))[0]) for t in xticks]

    curves_px = []
    for xd, yd in curves_data:
        pts = ax.transData.transform(np.column_stack([xd, yd]))
        curves_px.append(np.column_stack([pts[:, 0], H - pts[:, 1]]))

    rgba = np.asarray(fig.canvas.buffer_rgba())
    img_bgr = rgba[..., [2, 1, 0]].copy()
    plt.close(fig)

    return SyntheticFigure(
        img_bgr=img_bgr,
        x_axis_row=H - float(bbox.y0),
        y_axis_col=float(bbox.x0),
        xtick_px=xtick_px,
        xtick_values=[float(t) for t in xticks],
        curves_px=curves_px,
        curves_data=curves_data,
        colors_rgb=_COLORS[:n_curves],
        xlim=xlim,
    )


def make_crossing_mask(
    shape: tuple[int, int] = (400, 600), thickness: int = 2
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Binary mask with two straight curves of the SAME color crossing in
    the middle -- the canonical tracing test. Returns (mask, [truth1, truth2])
    where each truth is (W, 2) of (col, row_float).
    """
    h, w = shape
    cols = np.arange(w, dtype=float)
    y1 = 0.35 * cols + 0.15 * h            # ascending
    y2 = -0.35 * cols + 0.75 * h           # descending; cross mid-image
    mask = np.zeros(shape, dtype=bool)
    for y in (y1, y2):
        for c in range(w):
            r = int(round(y[c]))
            if 0 <= r < h:
                mask[max(r - thickness // 2, 0) : min(r + thickness // 2 + 1, h), c] = True
    return mask, [np.column_stack([cols, y1]), np.column_stack([cols, y2])]
