"""Export and re-plotting utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import Spectrum


def to_csv(spectrum: Spectrum, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([spectrum.x_label or "x", spectrum.y_label or "y"])
        for xi, yi in zip(spectrum.x, spectrum.y):
            writer.writerow([xi, yi])


def to_json(spectra: list[Spectrum], path: Path) -> None:
    path.write_text(json.dumps([s.to_json_dict() for s in spectra], indent=2))


def from_json(path: Path) -> list[Spectrum]:
    import numpy as np

    out = []
    for d in json.loads(path.read_text()):
        out.append(Spectrum(
            x=np.asarray(d["x"]), y=np.asarray(d["y"]),
            x_label=d.get("x_label"), y_label=d.get("y_label"),
            curve_label=d.get("curve_label"), provenance=d.get("provenance", {}),
        ))
    return out


def replot(spectra: list[Spectrum], path: Path,
           size_px: tuple[int, int] | None = None, dpi: int = 100) -> None:
    """Re-render extracted spectra. Doubles as the verify loop's RENDER step:
    pass size_px = source figure size to get a comparable image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figsize = (size_px[0] / dpi, size_px[1] / dpi) if size_px else (8, 5)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for s in spectra:
        color = None
        rgb = s.provenance.get("color_rgb")
        if rgb:
            color = tuple(c / 255 for c in rgb)
        ax.plot(s.x, s.y, label=s.curve_label, color=color)
    if spectra:
        ax.set_xlabel(spectra[0].x_label or "")
        ax.set_ylabel(spectra[0].y_label or "")
    if any(s.curve_label for s in spectra):
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
