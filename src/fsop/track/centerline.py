"""Polyline geometry: resampling, headings, normals, curvature, projection."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _wrap_diff(p: np.ndarray, closed: bool) -> np.ndarray:
  if closed:
    return np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
  d = np.gradient(p, axis=0)
  return d


def headings(p: np.ndarray, closed: bool) -> np.ndarray:
  d = _wrap_diff(p, closed)
  return np.arctan2(d[:, 1], d[:, 0])


def normals(p: np.ndarray, closed: bool) -> np.ndarray:
  """Unit left normals."""
  h = headings(p, closed)
  return np.stack([-np.sin(h), np.cos(h)], axis=1)


def curvature(p: np.ndarray, closed: bool) -> np.ndarray:
  """Signed curvature (positive = left turn) via finite differences on an (approx.) uniform polyline."""
  if closed:
    d1 = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / 2
    d2 = np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)
  else:
    d1 = np.gradient(p, axis=0)
    d2 = np.gradient(d1, axis=0)
  num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
  den = np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12)
  return num / den


def resample(p: np.ndarray, ds: float, closed: bool) -> np.ndarray:
  """Resample a polyline at uniform arc length ``ds`` using a (periodic) cubic spline."""
  p = np.asarray(p, dtype=np.float64)
  if closed and np.linalg.norm(p[0] - p[-1]) < 1e-9:
    p = p[:-1]
  pts = np.vstack([p, p[:1]]) if closed else p
  seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
  keep = np.concatenate([[True], seg > 1e-9])
  pts = pts[keep]
  t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
  if closed:
    pts[-1] = pts[0]
  cs = CubicSpline(t, pts, bc_type="periodic" if closed else "natural")
  # Two passes: spline arc length differs from chord length.
  for _ in range(2):
    tf = np.linspace(0, t[-1], max(int(t[-1] / (ds / 4)), 16))
    pf = cs(tf)
    sf = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pf, axis=0), axis=1))])
    total = sf[-1]
  n = max(int(round(total / ds)), 4)
  s_target = np.arange(n) * (total / n) if closed else np.linspace(0, total, n + 1)
  t_target = np.interp(s_target, sf, tf)
  return cs(t_target)


def arclength(p: np.ndarray, closed: bool) -> np.ndarray:
  s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
  return s


def total_length(p: np.ndarray, closed: bool) -> float:
  s = arclength(p, closed)[-1]
  if closed:
    s += np.linalg.norm(p[0] - p[-1])
  return float(s)


def project(p: np.ndarray, closed: bool, xy: np.ndarray, hint: int | None = None, window: int = 200):
  """Project point ``xy`` onto the polyline.

  Returns (index of segment start, fraction along segment, signed lateral offset (left +), s).
  ``hint`` limits the search to a window around a previous index for O(1) tracking.
  """
  n = len(p)
  if hint is None:
    idx = np.arange(n if closed else n - 1)
  else:
    idx = np.arange(hint - window // 4, hint + window) % n if closed else np.clip(
      np.arange(hint - window // 4, hint + window), 0, n - 2)
  a = p[idx]
  b = p[(idx + 1) % n]
  ab = b - a
  L2 = np.maximum((ab**2).sum(1), 1e-12)
  t = np.clip(((xy - a) * ab).sum(1) / L2, 0, 1)
  proj = a + ab * t[:, None]
  d2 = ((xy - proj) ** 2).sum(1)
  k = int(np.argmin(d2))
  i = int(idx[k])
  seg = ab[k]
  cross = seg[0] * (xy[1] - a[k][1]) - seg[1] * (xy[0] - a[k][0])
  lat = np.sign(cross) * np.sqrt(d2[k])
  s_arr = arclength(p, closed)
  s = s_arr[i] + t[k] * np.sqrt(L2[k])
  return i, float(t[k]), float(lat), float(s)


def straight_runs(kappa: np.ndarray, ds: float, closed: bool, kappa_straight: float = 1 / 200) -> float:
  """Longest contiguous run where |kappa| < kappa_straight (wraps for closed tracks)."""
  mask = np.abs(kappa) < kappa_straight
  if mask.all():
    return len(mask) * ds
  if closed:
    # rotate so we start on a curve
    start = int(np.argmin(mask))
    mask = np.roll(mask, -start)
  best = cur = 0
  for m in mask:
    cur = cur + 1 if m else 0
    best = max(best, cur)
  return best * ds
