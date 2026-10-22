"""Fixed-geometry disciplines: skidpad (D 4.1) and acceleration (D 5.1).

World frame: the car drives along +x, +y is left. ``start_pose`` is the vehicle reference point
(centre of gravity); REF_TO_FRONT converts rule staging distances (given for the vehicle front).
"""

from __future__ import annotations

import numpy as np

from fsop.track import centerline as cl
from fsop.track.rules import RULES
from fsop.track.schema import Cone, ConeColor, Track, TrackStats

DS = 0.25
REF_TO_FRONT = 1.6  # [m] CoG -> front-most point of a typical FS car


def _gate(x: float, half_w: float, gap: float = 0.6) -> list[Cone]:
  return [Cone(x + dx, sy * half_w, ConeColor.BIG_ORANGE, "start_gate") for dx in (-gap, gap) for sy in (1, -1)]


def _finish_stats(track: Track) -> Track:
  for i, c in enumerate(track.cones):
    c.id = i
  kappa = track.curvature()
  track.stats = TrackStats(
    length=float(cl.total_length(track.centerline, track.closed)),
    min_radius=float(1 / max(np.abs(kappa).max(), 1e-9)),
    max_straight=0.0,
    min_width=float((track.width_left + track.width_right).min()),
    n_cones=len(track.cones),
  )
  return track


def acceleration(width: float = 3.0, seed: int = 0) -> Track:
  L = RULES.accel_length.value
  stop = RULES.accel_stop_zone.value
  spacing = RULES.accel_cone_spacing.value
  half = width / 2
  x = np.arange(-20.0, L + stop + 10.0 + 1e-9, DS)
  center = np.stack([x, np.zeros_like(x)], 1)
  cones: list[Cone] = []
  for xi in np.arange(0.0, L + 1e-9, spacing):
    cones.append(Cone(float(xi), half, ConeColor.BLUE, "left"))
    cones.append(Cone(float(xi), -half, ConeColor.YELLOW, "right"))
  # staging area behind the line + stop zone
  for xi in np.arange(-10.0, 0.0, spacing):
    cones += [Cone(float(xi), half, ConeColor.ORANGE, "marking"), Cone(float(xi), -half, ConeColor.ORANGE, "marking")]
  for xi in np.arange(L + spacing, L + stop + 1e-9, spacing):
    cones += [Cone(float(xi), half, ConeColor.ORANGE, "stop_zone"),
              Cone(float(xi), -half, ConeColor.ORANGE, "stop_zone")]
  gates = _gate(0.0, half) + _gate(L, half)
  g = np.array([[c.x, c.y] for c in gates])
  cones = [c for c in cones if np.min(np.linalg.norm(g - [c.x, c.y], axis=1)) > 1.0] + gates
  t = Track(
    id=f"acceleration_w{width:.1f}",
    discipline="acceleration",
    seed=seed,
    closed=False,
    centerline=center,
    width_left=np.full(len(x), half),
    width_right=np.full(len(x), half),
    cones=cones,
    start_pose=(-RULES.accel_staging.value - REF_TO_FRONT, 0.0, 0.0),
    timekeeping_line=((0.0, half), (0.0, -half)),
    finish_line=((L, half), (L, -half)),
    stop_zone=[(L, half), (L + stop, half), (L + stop, -half), (L, -half)],
    laps=1,
    generator={"length": L, "width": width},
  )
  return _finish_stats(t)


def skidpad(seed: int = 0) -> Track:
  R_in = RULES.skidpad_inner_diameter.value / 2
  R_out = RULES.skidpad_outer_diameter.value / 2
  dc = RULES.skidpad_center_distance.value
  R = (R_in + R_out) / 2  # driving-line radius (= dc / 2: circles touch at the origin)
  c_right = np.array([0.0, -dc / 2])
  c_left = np.array([0.0, dc / 2])
  half = (R_out - R_in) / 2
  cones: list[Cone] = []
  # inner circles: 17 cones each, full circle
  for center, inner_color, tag in ((c_right, ConeColor.YELLOW, "right"), (c_left, ConeColor.BLUE, "left")):
    for a in np.linspace(0, 2 * np.pi, RULES.skidpad_inner_cones, endpoint=False):
      cones.append(Cone(*(center + R_in * np.array([np.cos(a), np.sin(a)])), inner_color, f"skidpad_inner_{tag}"))
  # outer circles: 13 cones over the arc that lies outside the other circle and away from the lanes
  for center, other, outer_color, tag in ((c_right, c_left, ConeColor.BLUE, "right"),
                                           (c_left, c_right, ConeColor.YELLOW, "left")):
    angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    pts = center + R_out * np.stack([np.cos(angles), np.sin(angles)], 1)
    ok = (np.linalg.norm(pts - other, axis=1) > R_out + 0.5) & (np.abs(pts[:, 1]) > half + 1.0)
    a_ok = angles[ok]
    # the valid region is one contiguous arc (possibly wrapping) on the far side
    ref = np.arctan2(*(center - other)[::-1])
    rel = np.angle(np.exp(1j * (a_ok - ref)))
    lo, hi = rel.min(), rel.max()
    for a in np.linspace(lo, hi, RULES.skidpad_outer_cones):
      cones.append(Cone(*(center + R_out * np.array([np.cos(ref + a), np.sin(ref + a)])), outer_color,
                        f"skidpad_outer_{tag}"))
  # entry / exit lanes (orange), 3 m wide
  staging = RULES.skidpad_staging.value
  stop = RULES.skidpad_stop_zone.value
  for xi in np.arange(-staging, -R_out + 0.1, 3.0):
    cones += [Cone(float(xi), half, ConeColor.ORANGE, "marking"), Cone(float(xi), -half, ConeColor.ORANGE, "marking")]
  for xi in np.arange(R_out, R_out + stop + 1e-9, 3.0):
    cones += [Cone(float(xi), half, ConeColor.ORANGE, "stop_zone"), Cone(float(xi), -half, ConeColor.ORANGE,
                                                                          "stop_zone")]
  # timekeeping line (x = 0): two big orange cones per side, just inside the inner circles
  cones += [Cone(dx, sy * (half + 0.3), ConeColor.BIG_ORANGE, "start_gate") for sy in (1, -1) for dx in (-0.6, 0.6)]
  # driving line: entry, right x2 (clockwise), left x2 (counter-clockwise), exit
  entry = np.stack([np.arange(-staging - 5, 0.0, DS), np.zeros(int(np.ceil((staging + 5) / DS)))], 1)
  n_circ = int(round(2 * np.pi * R / DS))
  phi = np.arange(2 * n_circ) * (2 * np.pi / n_circ)
  right = c_right + R * np.stack([np.sin(phi), np.cos(phi)], 1)
  left = c_left + R * np.stack([np.sin(phi), -np.cos(phi)], 1)
  exit_ = np.stack([np.arange(0.0, R_out + stop + 10, DS), np.zeros(int(np.ceil((R_out + stop + 10) / DS)))], 1)
  center = np.vstack([entry, right, left, exit_])
  n_entry, n_c = len(entry), len(right)
  route = ["entry"] + ["right"] * 2 + ["left"] * 2 + ["exit"]
  t = Track(
    id="skidpad",
    discipline="skidpad",
    seed=seed,
    closed=False,
    centerline=center,
    width_left=np.full(len(center), half),
    width_right=np.full(len(center), half),
    cones=cones,
    start_pose=(-staging - REF_TO_FRONT, 0.0, 0.0),
    timekeeping_line=((0.0, half), (0.0, -half)),
    finish_line=((0.0, half), (0.0, -half)),
    stop_zone=[(R_out, half), (R_out + stop, half), (R_out + stop, -half), (R_out, -half)],
    route=route,
    laps=4,
    generator={"center_distance": dc, "R_in": R_in, "R_out": R_out, "R_drive": R,
               "segments": {"entry": [0, n_entry], "right": [n_entry, n_entry + n_c],
                            "left": [n_entry + n_c, n_entry + 2 * n_c], "exit": [n_entry + 2 * n_c, len(center)]}},
  )
  return _finish_stats(t)
