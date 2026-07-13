"""Claude vision perception layer.

Design rules:
  1. Every call is a pure function of (model, prompt, images) -> JSON.
     Structured output is enforced by prompting for strict JSON and
     parsing defensively.
  2. Every response is cached on disk keyed by sha256(model|prompt|image
     bytes). API sampling is not bit-reproducible even at temperature 0,
     so the cache is what makes pipeline reruns deterministic (and free).
  3. If no API key is configured, ClaudeVision runs in *offline mode*:
     calls raise OfflineError. Pipeline stages catch this and degrade
     (e.g. segmentation falls back to un-primed k-means). Tests never
     need network access.

Claude's role is strictly SEMANTIC: reading labels/values/captions,
classifying, counting. Anything requiring pixel precision is done by
classical CV in axes.py / segmentation.py — vision transformers localize
at patch granularity, not pixel granularity.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from .config import Config


class OfflineError(RuntimeError):
    """Raised when a Claude call is attempted without an API key."""


# ----------------------------------------------------------------------
# Prompt templates. Keep them here (not inline) so they are diffable and
# unit-testable. Each asks for strict JSON with an explicit schema.
# ----------------------------------------------------------------------

PROMPT_FIND_FIGURES = """\
This is a rendered page from a scientific paper. Identify every figure on
the page (plots, micrographs, schematics -- not tables, not equations).
Return STRICT JSON, no prose:
{"figures": [{"bbox": [x0, y0, x1, y1], "kind": "line_plot|micrograph|schematic|other",
              "looks_like_spectrum": true|false}]}
bbox is in pixels of this image, origin top-left. Image size: {width}x{height}.
"""

PROMPT_CLASSIFY_FIGURE = """\
This image is a figure from a materials-science paper. Return STRICT JSON:
{"is_spectrum_plot": true|false,
 "plot_kind": "raman|xanes|xrd|absorption|pl|ftir|other|not_a_plot",
 "n_panels": <int>,
 "panels": [{"bbox": [x0, y0, x1, y1], "label": "(a)" or null,
             "is_spectrum_plot": true|false}]}
A "spectrum plot" is a line graph of intensity/absorption vs. an
energy-like axis. If the figure is a single panel, n_panels is 1 and
panels has one entry covering the whole image.
"""

PROMPT_READ_TICKS = """\
This image is a horizontal strip containing the tick labels of the {axis}-axis
of a plot (and possibly the axis title). Read it and return STRICT JSON:
{"tick_values": [<numbers, in the order they appear left-to-right or top-to-bottom>],
 "axis_label": "<axis title with units, or null>",
 "scale": "linear|log|unknown"}
Only include actual tick VALUES (numbers printed at tick marks), not the
axis title. Judge "log" from the spacing/values (e.g. 1, 10, 100).
"""

PROMPT_CURVE_PRIORS = """\
This image is the interior of a line plot containing one or more spectra.
Return STRICT JSON:
{"n_curves": <int>,
 "colors": [{"hex": "#rrggbb", "n_curves_this_color": <int>, "legend_label": "<or null>"}],
 "has_legend": true|false,
 "curves_share_colors": true|false,
 "vertically_offset_stack": true|false}
Count curves carefully -- offset-stacked spectra of the same color count
individually.
"""

PROMPT_READ_CAPTION = """\
This image shows a figure and the text below it from a paper page.
Return STRICT JSON: {"caption": "<the full figure caption, or null>"}
"""


# ----------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------

class ClaudeVision:
    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()
        self._client = None  # lazy: anthropic import only when actually used

    @property
    def online(self) -> bool:
        return bool(self.config.api_key)

    # -- public perception calls ---------------------------------------

    def find_figures(self, page_png: bytes, width: int, height: int) -> dict:
        prompt = PROMPT_FIND_FIGURES.replace("{width}", str(width)).replace("{height}", str(height))
        return self.ask_json(prompt, [page_png])

    def classify_figure(self, image_png: bytes) -> dict:
        return self.ask_json(PROMPT_CLASSIFY_FIGURE, [image_png])

    def read_ticks(self, strip_png: bytes, axis: str = "x") -> dict:
        return self.ask_json(PROMPT_READ_TICKS.replace("{axis}", axis), [strip_png])

    def curve_priors(self, plot_png: bytes) -> dict:
        return self.ask_json(PROMPT_CURVE_PRIORS, [plot_png])

    def read_caption(self, region_png: bytes) -> dict:
        return self.ask_json(PROMPT_READ_CAPTION, [region_png])

    # -- machinery ------------------------------------------------------

    def ask_json(self, prompt: str, images: list[bytes]) -> dict:
        """One cached, JSON-returning vision call."""
        key = self._cache_key(prompt, images)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if not self.online:
            raise OfflineError(
                "ANTHROPIC_API_KEY not set and result not in cache. "
                "Set the key in .env, or run stages that don't need vision."
            )
        raw = self._call_api(prompt, images)
        parsed = self._parse_json(raw)
        self._cache_put(key, parsed)
        return parsed

    def _call_api(self, prompt: str, images: list[bytes]) -> str:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.config.api_key)
        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(img).decode(),
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": prompt})
        resp = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Parse model output into JSON, tolerating code fences / preamble."""
        text = raw.strip()
        # strip ```json ... ``` fences if present
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # fall back to the outermost {...} span
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON object in model response: {raw[:200]!r}")
            text = text[start : end + 1]
        return json.loads(text)

    # -- disk cache ------------------------------------------------------

    def _cache_key(self, prompt: str, images: list[bytes]) -> str:
        h = hashlib.sha256()
        h.update(self.config.model.encode())
        h.update(prompt.encode())
        for img in images:
            h.update(hashlib.sha256(img).digest())
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _cache_put(self, key: str, value: dict) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(key).write_text(json.dumps(value, indent=1))
