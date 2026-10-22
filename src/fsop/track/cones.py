"""Cone placement along track boundaries."""

from __future__ import annotations

import numpy as np

from fsop.track import centerline as cl
from fsop.track.schema import Cone, ConeColor, Track

SIDE_COLOR = {"left": ConeColor.BLUE, "right": ConeColor.YELLOW}


def place_boundary_cones(boundary: np.ndarray, side: str, closed: bool, cfg) -> list[Cone]:
  """Curvature-adaptive cone spacing: dense in corners, at most ``cone_spacing_max`` on straights."""
  b = cl.resample(boundary, 0.1, closed)
  kappa = np.abs(cl.curvature(b, closed))
  s = cl.arclength(b, closed)
  L = cl.total_length(b, closed)
  # look-ahead minimum radius over the next few metres so spacing tightens before the corner
  radius = 1 / np.maximum(kappa, 1e-6)
  win = int(3.0 / 0.1)
  if closed:
    ext = np.concatenate([radius, radius[:win]])
  else:
    ext = np.concatenate([radius, np.full(win, radius[-1])])
  r_ahead = np.array([ext[i : i + win].min() for i in range(len(radius))])
  spacing = np.clip(cfg.cone_spacing_k * r_ahead, cfg.cone_spacing_min, cfg.cone_spacing_max)
  cones = []
  pos = 0.0
  end = L - 0.5 * cfg.cone_spacing_min if closed else L + 1e-6
  while pos < end:
    i = min(int(np.searchsorted(s, pos)), len(b) - 1)
    cones.append(Cone(float(b[i, 0]), float(b[i, 1]), SIDE_COLOR[side], side))
    step = float(spacing[i])
    if closed and pos + step > L - cfg.cone_spacing_min and pos + step < L:
      # avoid an awkward tiny gap at the seam: spread the remainder
      step = (L - pos) / 2 if L - pos > 2 * cfg.cone_spacing_min else L - pos
    pos += step
  return cones


def add_start_gate(track: Track, s_line: float = 0.0, gap: float = 0.6, clearance: float = 1.2) -> None:
  """Big orange cones (2 per side) at the timekeeping line; removes boundary cones too close to them."""
  p = track.centerline
  n = len(p)
  ds = track.length / n if track.closed else track.s[-1] / (n - 1)
  left, right = track.boundaries()
  gate = []
  for off in (-gap, gap):
    i = int(round((s_line + off) / ds)) % n
    gate.append(Cone(float(left[i, 0]), float(left[i, 1]), ConeColor.BIG_ORANGE, "start_gate"))
    gate.append(Cone(float(right[i, 0]), float(right[i, 1]), ConeColor.BIG_ORANGE, "start_gate"))
  g = np.array([[c.x, c.y] for c in gate])
  kept = [c for c in track.cones if np.min(np.linalg.norm(g - np.array([c.x, c.y]), axis=1)) > clearance]
  track.cones = kept + gate
  i0 = int(round(s_line / ds)) % n
  track.timekeeping_line = (tuple(map(float, left[i0])), tuple(map(float, right[i0])))
  track.finish_line = track.timekeeping_line
