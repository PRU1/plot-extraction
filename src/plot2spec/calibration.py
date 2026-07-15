""" Maps pixel to (x,y) coordinate

Input: AxesGeometry
Output: calibration that maps pixel point to (x,y) point"""

from __future__ import annotations
from multiprocessing import Value
from numpy._typing._array_like import NDArray
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from .models import AxesGeometry

@dataclass
class AxisMap:
   """Linear map between pixel point and x,y point. of form a*px + b or 10^(a*px+b) for log plots"""
   kind: Literal["linear", "log", "normalized"]
   a: float
   b: float
   def to_data(self, px: np.ndarray) -> np.ndarray:
      lin = self.a * np.asarray(px, dtype=float) + self.b
      if self.kind == "log":
         return 10.0 ** lin
      else:
         return lin

@dataclass
class Calibration:
   x_map: AxisMap
   y_map: AxisMap
   report: dict = field(default_factory=dict)

   def px_to_data(self, points_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      return (self.x_map.to_data(points_px[:, 0]), self.y_map.to_data(points_px[:, 1]))


def _extract_useable(ticks: list) -> tuple[np.ndarray, np.ndarray]:
   px, val = [], []
   for t in ticks:
      p = getattr(t, "px", None)
      v = getattr(t, "value", None)
      if p is None and isinstance(t, (tuple, list)):  # allow raw (px, value) pairs too
         p, v = t[0], t[1]
      if v is None:
         continue
      px.append(float(p))
      val.append(float(v))
   return np.asarray(px, float), np.asarray(val, float)


def _least_square_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
   A = np.vstack([x, np.ones_like(x)]).T
   (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
   return float(a), float(b)

def _robust_line(x: np.ndarray, y: np.ndarray, k: float=3.5, max_iter: int=5) -> tuple[float, float, np.ndarray]:
   # improve linear fit by throwing away terms that are far away from an initial linear fit. then rerun the fit. then throw away some more terms, etc. etc.
   x = np.asarray(x, float)
   y = np.asarray(y, float)
   mask = np.ones(len(x), bool)
   a, b = _least_square_regression(x, y)
   y_scale = 1.0 + float(np.max(np.abs(y))) if len(y) else 1.0

   for _ in range(max_iter):
      a, b = _least_square_regression(x[mask], y[mask]) # np boolean mask
      r = y - (a*x + b)                                 # residuals for ALL points, not just inliers
      r_in = r[mask]
      # robust noise scale from inlier residuals (1.4826 * MAD ~= sigma for Gaussian) --> google MAD.  1.4826 is 1/zscore. here 75 percentile score is used
      sigma = 1.4826 * float(np.median(np.abs(r_in - np.median(r_in)))) 
      if sigma < 1e-9 * y_scale:   # essentially a perfect fit; nothing to reject
         break
      new = np.abs(r) <= k * sigma  # keep points within k robust std of the line
      if new.sum() < 2 or np.array_equal(new, mask):
         if new.sum() >= 2:
            mask: NDArray[numpy.bool[builtins.bool]] = new
         break                      # converged, or can't reject below 2 points
      mask = new

   return a, b, mask

def _median_rel_resid(v_pred: np.ndarray, v: np.ndarray) -> float:
   # helper to decide whether to use log or linear plot. calculates median of relative error for every data point
   # func later will choose smaller error
   denom = np.where(np.abs(v) > 1e-12, np.abs(v), 1e-12) # divide by 0 guard
   return float(np.median(np.abs(v_pred-v)/denom))


def fit_axis_map(ticks: list, scale_hint: str="unknown", diagnostics: dict | None = None) -> AxisMap:
   # fit one axis pixel->value map.
   px, v = _extract_useable(ticks)
   if len(px) < 2:
      raise ValueError("need more axis data points. Less than 2 given.")
   
   # test linear scale
   a_lin, b_lin, mask_lin = _robust_line(x=px, y=v)
   v_lin = a_lin*px + b_lin
   score_lin = _median_rel_resid(v_lin, v)

   # test log scale
   score_log = np.inf
   a_log = b_log = None
   mask_log = None
   if np.all(v > 0):
      a_log, b_log, mask_log = _robust_line(px, np.log10(v))
      v_log = 10.0 ** (a_log * px + b_log)
      score_log = _median_rel_resid(v_log, v)
   
   # pick smaller score

   if score_log < score_lin:
      kind, a, b, mask = "log", a_log, b_log, mask_log
      v_pred = 10.0 ** (a * px + b)
   else:
      kind, a, b, mask = "linear", a_lin, b_lin, mask_lin
      v_pred = a * px + b

   # claude generated code that I'll keep:
   if diagnostics is not None:
      resid = v_pred[mask] - v[mask]
      diagnostics.update(
         kind=kind, a=a, b=b,
         n_used=int(mask.sum()),
         n_rejected=int((~mask).sum()),
         resid_rms=float(np.sqrt(np.mean(resid ** 2))) if mask.any() else float("nan"),
      )
      warnings = diagnostics.setdefault("warnings", [])
      if scale_hint in ("linear", "log") and scale_hint != kind:
         warnings.append(f"scale hint was '{scale_hint}' but fit prefers '{kind}' "
                         f"(rel-resid linear={score_lin:.2e}, log={score_log:.2e})")

   return AxisMap(kind=kind, a=float(a), b=float(b))
      

def _normalized_y(geometry: AxesGeometry) -> AxisMap:
   # claude generated func that's supposed to be failsafe in case y ticks are not useable. it normalizes the y axis based on end points
   bottom = float(geometry.x_axis_row)
   top = float(geometry.plot_bbox.y0)
   if abs(top - bottom) < 1e-9:
      raise ValueError("degenerate plot box: top row == x_axis_row")
   a = 1.0 / (top - bottom)   # negative, since top row index < bottom
   b = -a * bottom
   return AxisMap(kind="normalized", a=a, b=b)


def calibrate(geometry: AxesGeometry) -> Calibration:
   # build Calibration from AxesGeometry
   report: dict = {"warnings": []}

   x_diag: dict = {}
   x_map = fit_axis_map(geometry.x_ticks, geometry.x_scale, x_diag)
   report["x"] = x_diag
   report["warnings"].extend(x_diag.get("warnings", []))

   y_diag: dict = {}
   try:
      y_map = fit_axis_map(geometry.y_ticks, geometry.y_scale, diagnostics=y_diag)
      report["y"] = y_diag
      report["warnings"].extend(y_diag.get("warnings", []))
   except ValueError as e:
      y_map = _normalized_y(geometry)
      report["y"] = {"kind": "normalized", "reason": str(e)}
      report["warnings"].append("unable to calibrate y axis from ticks, used normalized [0,1] map")
   
   return Calibration(x_map, y_map, report)
