"""Stage A: PDF -> figure crops (+ captions, classification, panels).

Strategy (in order of preference):

  1. EMBEDDED extraction (lossless). Raster figures live in the PDF as
     image XObjects; pymupdf hands them back at native resolution with
     their placement rectangle. No rendering, no resampling.
  2. RENDER + Claude fallback. Vector figures (most matplotlib/Origin
     plots!) have no XObject to extract -- they are drawing commands.
     We render the page and ask Claude for figure bounding boxes, padded
     to absorb bbox imprecision, then crop at high DPI.
  3. Candidates from both paths are deduped by IoU of their page rects.

Captions: tried from the PDF *text layer* first (free and exact) -- the
text block directly below a figure rect that starts with "Fig"/"Figure".
Claude vision is the fallback for image-only (scanned) pages.

Classification & panel splitting: one Claude call per candidate decides
spectrum-or-not, plot kind, and panel sub-boxes; panels are cropped into
their own files so the extraction stage always sees a single plot.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import fitz  # pymupdf

from .claude import ClaudeVision, OfflineError
from .config import Config
from .models import BBox, FigureCandidate, Panel

_CAPTION_RE = re.compile(r"^\s*(fig(ure)?\.?|scheme)\s*\d+", re.IGNORECASE)


# ----------------------------------------------------------------------
# 1. Embedded (lossless) extraction
# ----------------------------------------------------------------------

def extract_embedded_images(doc: fitz.Document, outdir: Path, cfg: Config) -> list[FigureCandidate]:
    candidates: list[FigureCandidate] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, info in enumerate(page.get_images(full=True)):
            xref = info[0]
            try:
                rect = page.get_image_bbox(info)
            except ValueError:
                rect = None
            pix = fitz.Pixmap(doc, xref)
            if pix.width < cfg.min_figure_px and pix.height < cfg.min_figure_px:
                continue  # icons, logos, ornaments
            if pix.colorspace and pix.colorspace.n > 3:  # CMYK etc. -> RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            path = outdir / f"p{page_index:03d}_embedded_{img_index}.png"
            pix.save(path)
            bbox = None
            if rect is not None:
                # keep the placement rect in *rendered-page* pixel coords for dedupe
                scale = cfg.render_dpi / 72.0
                bbox = BBox(rect.x0 * scale, rect.y0 * scale, rect.x1 * scale, rect.y1 * scale)
            candidates.append(
                FigureCandidate(image_path=path, source="embedded", page_index=page_index, bbox_on_page=bbox)
            )
    return candidates


# ----------------------------------------------------------------------
# 2. Render + Claude bbox fallback
# ----------------------------------------------------------------------

def render_page_png(page: fitz.Page, dpi: int) -> tuple[bytes, int, int]:
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png"), pix.width, pix.height


def locate_rendered_figures(
    doc: fitz.Document, outdir: Path, cfg: Config, vision: ClaudeVision,
    existing: list[FigureCandidate],
) -> list[FigureCandidate]:
    """Ask Claude for figure bboxes on each rendered page; crop the new ones."""
    from PIL import Image  # pillow ships with matplotlib's dep tree

    candidates: list[FigureCandidate] = []
    for page_index in range(len(doc)):
        png, w, h = render_page_png(doc[page_index], cfg.render_dpi)
        try:
            result = vision.find_figures(png, w, h)
        except OfflineError:
            break  # offline: embedded-only ingest
        page_img = Image.open(io.BytesIO(png))
        page_bounds = BBox(0, 0, w, h)
        for i, fig in enumerate(result.get("figures", [])):
            bbox = BBox(*fig["bbox"]).padded(cfg.bbox_pad_px, bounds=page_bounds)
            # dedupe against embedded extractions on the same page
            dup = any(
                c.page_index == page_index
                and c.bbox_on_page is not None
                and c.bbox_on_page.iou(bbox) > cfg.iou_dedupe_threshold
                for c in existing
            )
            if dup:
                continue
            path = outdir / f"p{page_index:03d}_render_{i}.png"
            page_img.crop(bbox.as_int_tuple()).save(path)
            candidates.append(
                FigureCandidate(image_path=path, source="render", page_index=page_index, bbox_on_page=bbox)
            )
    return candidates


# ----------------------------------------------------------------------
# 3. Captions from the text layer (Claude fallback)
# ----------------------------------------------------------------------

def find_caption_textlayer(page: fitz.Page, bbox_render_px: BBox | None, cfg: Config) -> str | None:
    """Caption = nearest text block below the figure rect that looks like 'Figure N ...'."""
    if bbox_render_px is None:
        return None
    scale = 72.0 / cfg.render_dpi  # render px -> pdf points
    fig_bottom = bbox_render_px.y1 * scale
    fig_x0, fig_x1 = bbox_render_px.x0 * scale, bbox_render_px.x1 * scale
    best: tuple[float, str] | None = None
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if y0 < fig_bottom - 5:            # must start below the figure
            continue
        if x1 < fig_x0 or x0 > fig_x1:     # must overlap horizontally
            continue
        if not _CAPTION_RE.match(text):
            continue
        gap = y0 - fig_bottom
        if best is None or gap < best[0]:
            best = (gap, " ".join(text.split()))
    return best[1] if best else None


# ----------------------------------------------------------------------
# 4. Classification + panel splitting
# ----------------------------------------------------------------------

def classify_and_split(candidate: FigureCandidate, outdir: Path, vision: ClaudeVision) -> None:
    """Fill in is_spectrum_plot / plot_kind / panels in place."""
    from PIL import Image

    png = candidate.image_path.read_bytes()
    try:
        result = vision.classify_figure(png)
    except OfflineError:
        return  # leave classification fields as None
    candidate.is_spectrum_plot = bool(result.get("is_spectrum_plot"))
    candidate.plot_kind = result.get("plot_kind")
    panels = result.get("panels") or []
    if len(panels) <= 1:
        candidate.panels = [
            Panel(image_path=candidate.image_path, label=None,
                  is_spectrum_plot=candidate.is_spectrum_plot)
        ]
        return
    img = Image.open(candidate.image_path)
    bounds = BBox(0, 0, img.width, img.height)
    for j, p in enumerate(panels):
        bbox = BBox(*p["bbox"]).padded(4, bounds=bounds)
        label = p.get("label")
        path = outdir / f"{candidate.image_path.stem}_panel{j}.png"
        img.crop(bbox.as_int_tuple()).save(path)
        candidate.panels.append(
            Panel(image_path=path, label=label, is_spectrum_plot=p.get("is_spectrum_plot"))
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def ingest_pdf(pdf_path: Path, outdir: Path, cfg: Config | None = None,
               vision: ClaudeVision | None = None) -> list[FigureCandidate]:
    """PDF -> classified FigureCandidates with panels and captions.

    Writes a manifest (figures.json) alongside the crops so downstream
    stages / Claude Code agents can pick up where this left off.
    """
    cfg = cfg or Config.from_env()
    vision = vision or ClaudeVision(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)

    candidates = extract_embedded_images(doc, outdir, cfg)
    candidates += locate_rendered_figures(doc, outdir, cfg, vision, existing=candidates)

    for c in candidates:
        c.caption = find_caption_textlayer(doc[c.page_index], c.bbox_on_page, cfg)
        classify_and_split(c, outdir, vision)

    manifest = {
        "source_pdf": str(pdf_path),
        "figures": [c.to_json_dict() for c in candidates],
    }
    (outdir / "figures.json").write_text(json.dumps(manifest, indent=2))
    return candidates
