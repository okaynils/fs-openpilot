"""Classic geometric baseline: cones (ego frame) -> boundary pairing -> midline -> plan.

Used two ways in closed loop:
  * ``geometric_gt``    - visible ground-truth cones: upper bound of a perfect perception stack
  * ``geometric_model`` - the fine-tuned model's own BEV cone detections: perception -> classic planner
This shows how much the learned end-to-end plan adds over (or loses against) a modular pipeline.
"""

from __future__ import annotations

import numpy as np

from fsop.sim.expert import T_IDXS


def midline(cones_xy: np.ndarray, cls: np.ndarray, max_range: float = 25.0) -> np.ndarray | None:
  """Return midline points [N,2] ordered by arc length, or None."""
  m = (np.linalg.norm(cones_xy, axis=1) < max_range) & (cones_xy[:, 0] > -1.0)
  b = cones_xy[m & (cls == 0)]
  y = cones_xy[m & (cls == 1)]
  if len(b) == 0 and len(y) == 0:
    return None
  if len(b) == 0 or len(y) == 0:  # one side only: offset by half a nominal track width
    side, sign = (b, -1.0) if len(b) else (y, 1.0)
    side = side[np.argsort(side[:, 0])]
    mid = side + np.array([0.0, sign * 1.75])
  else:
    mids = []
    for p in b:
      q = y[np.argmin(np.linalg.norm(y - p, axis=1))]
      if np.linalg.norm(q - p) < 7.0:
        mids.append((p + q) / 2)
    for q in y:
      p = b[np.argmin(np.linalg.norm(b - q, axis=1))]
      if np.linalg.norm(q - p) < 7.0:
        mids.append((p + q) / 2)
    if not mids:
      return None
    mid = np.unique(np.round(np.array(mids), 2), axis=0)
  # greedy ordering from the car forward
  order, cur, rest = [], np.zeros(2), list(range(len(mid)))
  while rest:
    d = [np.linalg.norm(mid[i] - cur) for i in rest]
    j = rest.pop(int(np.argmin(d)))
    order.append(mid[j])
    cur = mid[j]
  return np.vstack([np.zeros(2), np.array(order)])


def plan_from_cones(cones_xy: np.ndarray, cls: np.ndarray, v: float, a_lat: float = 6.0, v_cap: float = 12.0):
  """Build a [33,15] ego-frame plan along the cone midline at a curvature-limited speed."""
  path = midline(cones_xy, cls)
  plan = np.zeros((33, 15))
  if path is None or len(path) < 2:
    plan[:, 0] = T_IDXS * max(v, 2.0)
    plan[:, 3] = max(v, 2.0)
    return plan
  seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
  s = np.concatenate([[0], np.cumsum(seg)])
  if s[-1] < 1.0:
    plan[:, 0] = T_IDXS * max(v, 2.0)
    return plan
  # curvature from heading change over the visible path
  h = np.unwrap(np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0])))
  kappa = np.abs(np.diff(h)) / np.maximum(seg[1:], 0.5) if len(h) > 1 else np.zeros(1)
  v_tgt = float(min(v_cap, np.sqrt(a_lat / max(kappa.max() if len(kappa) else 1e-3, 1e-3))))
  v_tgt = min(v_tgt, max(v, 2.0) + 3.0)  # bounded acceleration
  d = np.minimum(T_IDXS * v_tgt, s[-1])  # do not extrapolate beyond the visible midline
  plan[:, 0] = np.interp(d, s, path[:, 0])
  plan[:, 1] = np.interp(d, s, path[:, 1])
  plan[:, 3] = v_tgt
  hh = np.concatenate([[0.0], h])
  plan[:, 11] = np.interp(d, s, hh)
  return plan
