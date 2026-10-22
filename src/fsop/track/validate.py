"""Rule validator for generated / imported tracks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LinearRing, LineString, Point, Polygon

from fsop.track import centerline as cl
from fsop.track.rules import RULES
from fsop.track.schema import ConeColor, Track


@dataclass
class RuleCheck:
  name: str
  passed: bool
  value: float | str
  limit: str
  severity: str = "error"  # error | warning


@dataclass
class ValidationReport:
  track_id: str
  checks: list[RuleCheck] = field(default_factory=list)

  @property
  def passed(self) -> bool:
    return all(c.passed for c in self.checks if c.severity == "error")

  def failed(self) -> list[RuleCheck]:
    return [c for c in self.checks if not c.passed]

  def __str__(self) -> str:
    bad = ", ".join(f"{c.name}={c.value} ({c.limit})" for c in self.failed())
    return f"[{self.track_id}] {'PASS' if self.passed else 'FAIL'}" + (f": {bad}" if bad else "")


def _add(rep: ValidationReport, name: str, ok: bool, value, limit: str, severity: str = "error"):
  if isinstance(value, float):
    value = round(value, 3)
  rep.checks.append(RuleCheck(name, bool(ok), value, limit, severity))


def track_polygon(track: Track) -> Polygon:
  """Drivable area. Closed loops: outer boundary minus inner. Open layouts (may self-overlap, e.g.
  skidpad): centerline buffered by the (constant) half width."""
  if track.closed:
    left, right = track.boundaries()
    return Polygon(left, holes=[right]) if Polygon(left).contains(Polygon(right)) else Polygon(right, holes=[left])
  half = float(np.median(np.minimum(track.width_left, track.width_right)))
  return LineString(track.centerline).buffer(half, cap_style="flat")


def validate(track: Track) -> ValidationReport:
  if track.discipline in ("trackdrive", "autocross"):
    return _validate_loop(track)
  if track.discipline == "acceleration":
    return _validate_accel(track)
  if track.discipline == "skidpad":
    return _validate_skidpad(track)
  raise ValueError(track.discipline)


def _validate_loop(track: Track) -> ValidationReport:
  rep = ValidationReport(track.id)
  p = track.centerline
  ds = track.length / len(p)
  L = track.length
  if track.discipline == "trackdrive":
    _add(rep, "lap_length", RULES.trackdrive_lap_min.value <= L <= RULES.trackdrive_lap_max.value, L,
         f"[{RULES.trackdrive_lap_min.value}, {RULES.trackdrive_lap_max.value}] m (D 8.1.2)")
  else:
    _add(rep, "length", L < RULES.autocross_max_length.value, L, "< 1500 m (D 6.1.2)")
  kappa = track.curvature()
  max_straight = cl.straight_runs(kappa, ds, track.closed)
  _add(rep, "max_straight", max_straight <= RULES.max_straight.value, max_straight, "<= 80 m (D 8.1.1)")
  r_min = 1 / max(np.abs(kappa).max(), 1e-9)
  _add(rep, "min_center_radius", 2 * r_min >= RULES.min_turning_diameter.value, r_min,
       ">= 4.5 m, i.e. 9 m turning diameter (D 8.1.1)")
  w = track.width_left + track.width_right
  _add(rep, "min_width", w.min() >= RULES.min_track_width.value - 1e-6, float(w.min()), ">= 3 m (D 8.1.1)")
  left, right = track.boundaries()
  _add(rep, "centerline_simple", LinearRing(p).is_simple, "", "no self-intersection")
  _add(rep, "boundaries_simple", LinearRing(left).is_simple and LinearRing(right).is_simple, "",
       "no self-intersection")
  # clearance between non-adjacent sections
  step = max(int(1.0 / ds), 1)
  sub = p[::step]
  half = np.maximum(track.width_left, track.width_right)[::step]
  tree = cKDTree(sub)
  max_r = 2 * half.max() + RULES.min_section_clearance.value
  pairs = tree.query_pairs(max_r, output_type="ndarray")
  min_clear = np.inf
  if len(pairs):
    n = len(sub)
    di = np.abs(pairs[:, 0] - pairs[:, 1])
    arc = np.minimum(di, n - di) * step * ds
    far = arc > 25.0
    if far.any():
      pr = pairs[far]
      d = np.linalg.norm(sub[pr[:, 0]] - sub[pr[:, 1]], axis=1) - half[pr[:, 0]] - half[pr[:, 1]]
      min_clear = float(d.min())
  _add(rep, "section_clearance", min_clear >= RULES.min_section_clearance.value,
       min_clear if np.isfinite(min_clear) else "inf", ">= 2 m boundary gap (generator)")
  _check_cones(track, rep)
  # staging pose inside track
  poly = track_polygon(track)
  sx, sy, _ = track.start_pose
  _add(rep, "staging_inside", poly.contains(Point(sx, sy)), "", "start pose inside track")
  return rep


def _check_cones(track: Track, rep: ValidationReport) -> None:
  n_gate = sum(c.color == ConeColor.BIG_ORANGE for c in track.cones)
  _add(rep, "start_gate", n_gate >= 4, n_gate, ">= 4 big orange cones")
  wrong_side = 0
  hint = None
  for side, color, sign in (("left", ConeColor.BLUE, 1), ("right", ConeColor.YELLOW, -1)):
    side_cones = [c for c in track.cones if c.color == color]
    xy = np.array([[c.x, c.y] for c in side_cones])
    if len(xy) < 3:
      _add(rep, f"{side}_cones", False, len(xy), ">= 3")
      continue
    for q in xy:
      _, _, lat, _ = cl.project(track.centerline, track.closed, q, hint)
      if np.sign(lat) != sign:
        wrong_side += 1
    d = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]) if track.closed else xy, axis=0), axis=1)
    # gaps at the start gate are bridged by the big orange cones -> allow 2x there
    _add(rep, f"{side}_max_spacing", np.sort(d)[-3] <= RULES.max_cone_spacing.value + 0.3 if len(d) > 3 else True,
         float(np.sort(d)[-3]) if len(d) > 3 else 0.0, "<= 5 m (handbook)", "warning")
    _add(rep, f"{side}_min_spacing", d.min() >= RULES.min_cone_spacing_same_side.value - 1e-6, float(d.min()),
         ">= 1 m")
  _add(rep, "cone_colors_sides", wrong_side == 0, wrong_side, "blue left / yellow right")


def _validate_accel(track: Track) -> ValidationReport:
  rep = ValidationReport(track.id)
  _add(rep, "length", abs(track.generator.get("length", 75.0) - RULES.accel_length.value) < 1e-6,
       track.generator.get("length", 75.0), "75 m (D 5.1.1)")
  w = track.width_left + track.width_right
  _add(rep, "min_width", w.min() >= RULES.accel_min_width.value - 1e-6, float(w.min()), ">= 3 m (D 5.1.1)")
  _check_cones(track, rep)
  return rep


def _validate_skidpad(track: Track) -> ValidationReport:
  rep = ValidationReport(track.id)
  g = track.generator
  _add(rep, "center_distance", abs(g["center_distance"] - RULES.skidpad_center_distance.value) < 1e-6,
       g["center_distance"], "18.25 m (D 4.1.2)")
  inner = [c for c in track.cones if c.role.startswith("skidpad_inner")]
  outer = [c for c in track.cones if c.role.startswith("skidpad_outer")]
  _add(rep, "inner_cones", len(inner) == 2 * RULES.skidpad_inner_cones, len(inner), "17 per circle (D 4.1.3)")
  _add(rep, "outer_cones", len(outer) == 2 * RULES.skidpad_outer_cones, len(outer), "13 per circle (D 4.1.3)")
  ls = LineString(track.centerline)
  _add(rep, "path_length", ls.length > 4 * np.pi * 9.0, float(ls.length), "covers 4 laps")
  return rep
