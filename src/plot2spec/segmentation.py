"""Curve-pixel segmentation: which pixels belong to plot lines, by color.

Replaces the paper's semantic-segmentation CNN with a training-free
classical method, optionally primed by Claude:

  1. PRIORS (optional, Claude): number of curves, their approximate
     colors, whether several curves share a color (offset stacks).
  2. COLOR CLUSTERING: k-means in CIELAB. Lab is used because Euclidean
     distance there approximates perceptual color difference (that is
     the design goal of the Lab space), so "the red curve" forms one
     tight cluster even under anti-aliasing gradients. k = n_colors + 2
     (background + black axes/text).
  3. CLUSTER TRIAGE: the background cluster is the most populous, high-L
     low-chroma one. The axis/text cluster is near-black AND mostly
     outside/on the plot border. What remains are curve-color clusters.
  4. CLEANUP: median filter would erase 1-px lines, so cleanup is
     deliberately gentle -- small-component removal only. Curves are thin;
     aggressive morphology destroys exactly the signal we want.

Output: one CurveMask per curve COLOR. Splitting same-colored curves
(and re-joining curves through crossings) is the tracing stage's job --
segmentation has no notion of instance, by design.
"""

from __future__ import annotations

import numpy as np
import cv2

from .claude import ClaudeVision, OfflineError
from .config import Config
from .models import BBox, CurveMask


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _get_priors(plot_bgr: np.ndarray, vision: ClaudeVision | None) -> dict | None:
    if vision is None:
        return None
    try:
        ok, buf = cv2.imencode(".png", plot_bgr)
        return vision.curve_priors(buf.tobytes())
    except OfflineError:
        return None


def segment_curves(img_bgr: np.ndarray, plot_bbox: BBox,
                   cfg: Config | None = None,
                   vision: ClaudeVision | None = None) -> list[CurveMask]:
    """Return one CurveMask per detected curve color inside plot_bbox."""
    cfg = cfg or Config()
    x0, y0, x1, y1 = plot_bbox.as_int_tuple()
    # shave the border so axis/frame lines don't enter the clustering
    m = 3
    interior = img_bgr[y0 + m : y1 - m, x0 + m : x1 - m]
    if interior.size == 0:
        return []

    priors = _get_priors(interior, vision)
    n_colors = None
    expected_per_color: dict[tuple[int, int, int], int] = {}
    if priors:
        colors = priors.get("colors", [])
        n_colors = max(len(colors), 1)
        for c in colors:
            try:
                expected_per_color[_hex_to_rgb(c["hex"])] = int(c.get("n_curves_this_color", 1))
            except (KeyError, ValueError):
                pass
    k = (n_colors + 2) if n_colors else cfg.default_k

    lab = cv2.cvtColor(interior, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, attempts=5,
                                    flags=cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(interior.shape[:2])

    masks: list[CurveMask] = []
    counts = np.bincount(labels.ravel(), minlength=k)
    for ci in range(k):
        L, a, b = centers[ci]
        chroma = float(np.hypot(a - 128.0, b - 128.0))
        # background: bright and unsaturated
        if L / 255.0 > cfg.background_l_threshold and chroma < 10:
            continue
        # residual axis/gridline/text ink: near-black or near-gray with few pixels
        if counts[ci] < cfg.min_curve_pixels:
            continue
        cluster_mask_small = labels == ci

        # gentle cleanup: drop connected components below a size floor
        n_cc, cc = cv2.connectedComponents(cluster_mask_small.astype(np.uint8))
        keep = np.zeros_like(cluster_mask_small)
        for c_id in range(1, n_cc):
            comp = cc == c_id
            if comp.sum() >= cfg.min_curve_pixels:
                keep |= comp
        if not keep.any():
            continue

        # embed back into full-image coordinates
        full = np.zeros(img_bgr.shape[:2], dtype=bool)
        full[y0 + m : y1 - m, x0 + m : x1 - m] = keep

        center_bgr = cv2.cvtColor(
            centers[ci].reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR
        )[0, 0]
        color_rgb = (int(center_bgr[2]), int(center_bgr[1]), int(center_bgr[0]))

        # attach the Claude prior for how many curves share this color, if any
        n_expected = None
        if expected_per_color:
            nearest = min(expected_per_color,
                          key=lambda c: sum((c[j] - color_rgb[j]) ** 2 for j in range(3)))
            if sum((nearest[j] - color_rgb[j]) ** 2 for j in range(3)) < 60 ** 2:
                n_expected = expected_per_color[nearest]

        masks.append(CurveMask(mask=full, color_rgb=color_rgb, n_expected_curves=n_expected))

    return masks
