"""Runtime configuration.

Everything tunable lives here so the pipeline stages stay pure-ish
functions of (input, config). Values can be overridden by environment
variables (P2S_* / ANTHROPIC_API_KEY), optionally loaded from a .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines) so we avoid a dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Config:
    # --- Claude / API ---
    api_key: str | None = None
    model: str = "claude-sonnet-5"
    max_tokens: int = 2048
    cache_dir: Path = Path(".cache/claude")

    # --- Stage A: ingest ---
    render_dpi: int = 200           # page render resolution for the vision fallback
    min_figure_px: int = 120        # embedded images smaller than this (both dims) are ignored (logos, icons)
    bbox_pad_px: int = 8            # padding added around Claude-reported figure bboxes (bbox imprecision)
    iou_dedupe_threshold: float = 0.5  # embedded vs. rendered figure dedupe

    # --- Stage B: axes ---
    axis_search_frac: float = 0.45  # y-axis searched in left 45% of image, x-axis in bottom 45%
    min_axis_run_frac: float = 0.5  # an axis line must span >= this fraction of the plot dimension
    tick_strip_px: int = 10         # strip thickness (outside the plot box) scanned for tick marks
    tick_min_separation_px: int = 15

    # --- Stage B: segmentation ---
    default_k: int = 4              # k-means clusters when Claude gives no curve-count prior
    background_l_threshold: float = 0.92  # (L/255) above this + low chroma => background
    min_curve_pixels: int = 50      # clusters with fewer foreground pixels are discarded

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv()
        cfg = cls()
        cfg.api_key = os.environ.get("ANTHROPIC_API_KEY")
        cfg.model = os.environ.get("P2S_MODEL", cfg.model)
        cfg.cache_dir = Path(os.environ.get("P2S_CACHE_DIR", str(cfg.cache_dir)))
        cfg.render_dpi = int(os.environ.get("P2S_RENDER_DPI", cfg.render_dpi))
        return cfg
