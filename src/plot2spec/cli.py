"""CLI: the agent-facing surface of the pipeline.

    p2s ingest paper.pdf -o out/figures        # Stage A
    p2s extract out/figures/p003_render_0.png  # Stage B (one plot image)
    p2s run paper.pdf -o out/                  # end to end
    p2s replot out/spectra.json                # visual sanity check

Stages that hit your unimplemented stubs stop GRACEFULLY: intermediates
produced so far (axes.json, masks/*.png) are saved, and the message
names the module to implement. So the plumbing is testable today and
each stub you fill in extends how far `p2s extract` gets.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import typer

from .config import Config

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _json_default(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


@app.command()
def ingest(pdf: Path, outdir: Path = typer.Option(Path("out/figures"), "-o")):
    """Stage A: PDF -> classified figure crops + captions (figures.json manifest)."""
    from .ingest import ingest_pdf

    candidates = ingest_pdf(pdf, outdir, Config.from_env())
    n_spectra = sum(1 for c in candidates if c.is_spectrum_plot)
    typer.echo(f"{len(candidates)} figures extracted, {n_spectra} classified as spectrum plots.")
    typer.echo(f"Manifest: {outdir / 'figures.json'}")


@app.command()
def extract(image: Path, outdir: Path = typer.Option(Path("out/extract"), "-o")):
    """Stage B: plot image -> spectra. Stops gracefully at unimplemented stubs."""
    import cv2

    from .axes import read_axes
    from .claude import ClaudeVision
    from .segmentation import segment_curves

    cfg = Config.from_env()
    vision = ClaudeVision(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image))
    if img is None:
        raise typer.BadParameter(f"cannot read image: {image}")

    # -- axes + ticks (working) --
    geometry = read_axes(img, cfg, vision)
    (outdir / "axes.json").write_text(json.dumps(dataclasses.asdict(geometry),
                                                 default=_json_default, indent=2))
    typer.echo(f"axes: x-axis row {geometry.x_axis_row}, y-axis col {geometry.y_axis_col}, "
               f"{len(geometry.x_ticks)} x-ticks -> {outdir/'axes.json'}")

    # -- segmentation (working) --
    masks = segment_curves(img, geometry.plot_bbox, cfg, vision)
    for i, m in enumerate(masks):
        cv2.imwrite(str(outdir / f"mask_{i}_rgb{m.color_rgb}.png"),
                    (m.mask * 255).astype("uint8"))
    typer.echo(f"segmentation: {len(masks)} curve-color masks -> {outdir}/mask_*.png")

    # -- your math from here on --
    try:
        from .calibration import calibrate
        calibration = calibrate(geometry)
    except NotImplementedError as e:
        typer.echo(f"\nstopped: {e}\n-> implement src/plot2spec/calibration.py "
                   f"(intermediates saved in {outdir})")
        raise typer.Exit(0)

    try:
        from .tracing import trace_curves
        traces = trace_curves(masks, geometry.plot_bbox)
    except NotImplementedError as e:
        typer.echo(f"\nstopped: {e}\n-> implement src/plot2spec/tracing.py")
        raise typer.Exit(0)

    try:
        from .export import replot, to_json
        from .signal import trace_to_spectrum
        spectra = [trace_to_spectrum(t, calibration) for t in traces]
    except NotImplementedError as e:
        typer.echo(f"\nstopped: {e}\n-> implement src/plot2spec/signal.py")
        raise typer.Exit(0)

    for s in spectra:
        s.x_label, s.y_label = geometry.x_label, geometry.y_label
        s.provenance.setdefault("source_image", str(image))
    to_json(spectra, outdir / "spectra.json")
    replot(spectra, outdir / "reconstruction.png", size_px=(img.shape[1], img.shape[0]))
    typer.echo(f"{len(spectra)} spectra -> {outdir/'spectra.json'} "
               f"(visual check: {outdir/'reconstruction.png'})")


@app.command()
def run(pdf: Path, outdir: Path = typer.Option(Path("out"), "-o")):
    """End to end: ingest, then extract every spectrum panel."""
    from .ingest import ingest_pdf

    cfg = Config.from_env()
    candidates = ingest_pdf(pdf, outdir / "figures", cfg)
    n = 0
    for c in candidates:
        for p in c.panels:
            if p.is_spectrum_plot or (p.is_spectrum_plot is None and c.is_spectrum_plot):
                extract(p.image_path, outdir / "extract" / p.image_path.stem)
                n += 1
    typer.echo(f"processed {n} spectrum panels.")


@app.command()
def replot_cmd(spectra_json: Path, out: Path = typer.Option(Path("replot.png"), "-o")):
    """Re-render extracted spectra for visual comparison."""
    from .export import from_json, replot

    replot(from_json(spectra_json), out)
    typer.echo(f"-> {out}")


if __name__ == "__main__":
    app()
