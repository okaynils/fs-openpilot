"""Track IO: canonical JSON, plus cone CSVs compatible with EUFS sim / FSDS track files.

EUFS (eufs_sim ``tracks/*.csv``): header ``tag,x,y,direction,x_variance,y_variance,xy_covariance``,
tags ``blue|yellow|orange|big_orange|car_start``. FSDS uses the same row layout without a header.
Both formats are flagged to-verify against current upstream sample tracks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from fsop.track import centerline as cl
from fsop.track.schema import Cone, ConeColor, Track

EUFS_HEADER = ["tag", "x", "y", "direction", "x_variance", "y_variance", "xy_covariance"]


def save_json(track: Track, path: str | Path) -> None:
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
    json.dump(track.to_dict(), f)


def load_json(path: str | Path) -> Track:
  with open(path) as f:
    return Track.from_dict(json.load(f))


def to_cone_csv(track: Track, path: str | Path, fmt: str = "eufs") -> None:
  rows = []
  sx, sy, syaw = track.start_pose
  rows.append(["car_start", sx, sy, syaw, 0, 0, 0])
  for c in track.cones:
    tag = c.color.value if c.color != ConeColor.UNKNOWN else "orange"
    rows.append([tag, round(c.x, 4), round(c.y, 4), 0, 0.01, 0.01, 0])
  with open(path, "w", newline="") as f:
    w = csv.writer(f)
    if fmt == "eufs":
      w.writerow(EUFS_HEADER)
    w.writerows(rows)


def from_cone_csv(path: str | Path, discipline: str = "trackdrive", track_id: str | None = None) -> Track:
  """Import a cone CSV (EUFS or FSDS layout); reconstructs a centerline from blue/yellow pairs."""
  cones: list[Cone] = []
  start = (0.0, 0.0, 0.0)
  with open(path) as f:
    for row in csv.reader(f):
      if not row or row[0] == "tag":
        continue
      tag, x, y = row[0], float(row[1]), float(row[2])
      if tag == "car_start":
        start = (x, y, float(row[3]) if len(row) > 3 else 0.0)
        continue
      color = {"blue": ConeColor.BLUE, "yellow": ConeColor.YELLOW, "orange": ConeColor.ORANGE,
               "big_orange": ConeColor.BIG_ORANGE}.get(tag, ConeColor.UNKNOWN)
      role = {"blue": "left", "yellow": "right"}.get(tag, "marking")
      cones.append(Cone(x, y, color, role))
  center, wl, wr = centerline_from_cones(cones, start)
  t = Track(id=track_id or Path(path).stem, discipline=discipline, seed=-1, closed=True, centerline=center,
            width_left=wl, width_right=wr, cones=cones, start_pose=start, generator={"imported": str(path)})
  for i, c in enumerate(t.cones):
    c.id = i
  return t


def centerline_from_cones(cones: list[Cone], start: tuple[float, float, float]):
  """Order blue and yellow cones by nearest-neighbour walk from the start, match pairs, take midpoints."""
  blue = np.array([[c.x, c.y] for c in cones if c.color == ConeColor.BLUE])
  yellow = np.array([[c.x, c.y] for c in cones if c.color == ConeColor.YELLOW])

  def order(pts: np.ndarray) -> np.ndarray:
    rest = list(range(len(pts)))
    cur = int(np.argmin(np.linalg.norm(pts - np.array(start[:2]), axis=1)))
    out = [cur]
    rest.remove(cur)
    heading = np.array([np.cos(start[2]), np.sin(start[2])])
    while rest:
      d = pts[rest] - pts[cur]
      dist = np.linalg.norm(d, axis=1)
      fwd = d @ heading / np.maximum(dist, 1e-9)
      score = dist * (1.5 - 0.5 * fwd)
      k = int(np.argmin(score))
      nxt = rest[k]
      heading = (pts[nxt] - pts[cur]) / max(dist[k], 1e-9)
      cur = nxt
      out.append(cur)
      rest.remove(cur)
    return pts[out]

  b = cl.resample(order(blue), 0.25, True)
  y = cl.resample(order(yellow), 0.25, True)
  from scipy.spatial import cKDTree

  tree = cKDTree(y)
  _, j = tree.query(b)
  mid = cl.resample((b + y[j]) / 2, 0.25, True)
  tb, ty = cKDTree(b), cKDTree(y)
  wl, _ = tb.query(mid)
  wr, _ = ty.query(mid)
  return mid, wl, wr
