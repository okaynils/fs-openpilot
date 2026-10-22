"""Rule referee: timing-line crossings, laps, DOO cone hits, off-course, stop zone, DQ rules (FSG D 2.6,
D 4-8, D 10)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from fsop.sim.vehicle import VehicleParams, VehicleState
from fsop.track.rules import RULES
from fsop.track.schema import ConeColor, Track
from fsop.track.validate import track_polygon


@dataclass
class Event:
  t: float
  kind: str  # line_cross | lap | doo | off_course | back_on_course | finish | stopped | dnf | dq
  data: dict = field(default_factory=dict)


def _seg_intersect(p1, p2, q1, q2) -> bool:
  def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

  d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
  d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
  return (d1 * d2 < 0) and (d3 * d4 < 0)


class Referee:
  def __init__(self, track: Track, cones_xy: np.ndarray, cone_colors: list[ConeColor], vp: VehicleParams):
    self.track = track
    self.vp = vp
    self.cones_xy = np.asarray(cones_xy, dtype=np.float64)
    self.cone_r = np.array([RULES.large_cone.base / 2 if c == ConeColor.BIG_ORANGE else RULES.small_cone.base / 2
                            for c in cone_colors])
    self.down = np.zeros(len(self.cones_xy), dtype=bool)
    self.events: list[Event] = []
    self.crossings: list[float] = []  # times of timing-line crossings (forward)
    self.finish_crossings: list[float] = []
    self.poly = prep(self._course_polygon())
    self.off = False
    self.prev_xy = None
    self.stopped_since: float | None = None
    self.finished = False
    self.done = False
    self.result = "running"
    self.started = False
    self.n_doo = 0
    self.n_oc = 0
    self.max_speed = 0.0

  def _course_polygon(self) -> Polygon:
    poly = track_polygon(self.track)
    if self.track.stop_zone:
      poly = poly.union(Polygon(self.track.stop_zone))
    if not self.track.closed:  # staging area behind the start for open layouts
      x, y, yaw = self.track.start_pose
      d = np.array([np.cos(yaw), np.sin(yaw)])
      n = np.array([-d[1], d[0]])
      p0 = np.array([x, y]) - 5 * d
      poly = poly.union(Polygon([p0 + 2 * n, p0 - 2 * n, p0 + 10 * d - 2 * n, p0 + 10 * d + 2 * n]))
    return poly.buffer(0.05)

  # ------------------------------------------------------------------------------------------
  def update(self, st: VehicleState, mission_done: bool = False) -> list[Event]:
    ev: list[Event] = []
    xy = np.array([st.x, st.y])
    self.max_speed = max(self.max_speed, st.vx)
    c, s = np.cos(st.yaw), np.sin(st.yaw)
    R = np.array([[c, -s], [s, c]])
    if self.prev_xy is not None:
      for line, store, kind in ((self.track.timekeeping_line, self.crossings, "line_cross"),
                                (self.track.finish_line, self.finish_crossings, "finish_cross")):
        if line is None or (kind == "finish_cross" and self.track.finish_line == self.track.timekeeping_line):
          continue
        if _seg_intersect(self.prev_xy, xy, line[0], line[1]):
          store.append(st.t)
          ev.append(Event(st.t, kind, {"n": len(store)}))
    self.prev_xy = xy
    # cone hits: footprint polygon vs cone base circle (approx. by distance to rectangle)
    near = np.where(~self.down & (np.linalg.norm(self.cones_xy - xy, axis=1) < 2.5))[0]
    if len(near):
      loc = (self.cones_xy[near] - xy) @ R  # into ego frame
      f, r, w = self.vp.lf + self.vp.front_overhang, -(self.vp.lr + self.vp.rear_overhang), self.vp.width / 2
      dx = np.maximum(np.maximum(r - loc[:, 0], loc[:, 0] - f), 0)
      dy = np.maximum(np.abs(loc[:, 1]) - w, 0)
      hit = np.hypot(dx, dy) < self.cone_r[near]
      for i in near[hit]:
        self.down[i] = True
        self.n_doo += 1
        ev.append(Event(st.t, "doo", {"cone": int(i), "dir": float(st.yaw)}))
    # off course: all four wheels outside
    wheels = self.vp.wheels() @ R.T + xy
    outside = all(not self.poly.contains(Point(p)) for p in wheels)
    if outside and not self.off:
      self.off = True
      self.n_oc += 1
      ev.append(Event(st.t, "off_course", {}))
    elif not outside and self.off:
      self.off = False
      ev.append(Event(st.t, "back_on_course", {}))
    # standstill / finish
    if st.vx < 0.05:
      if self.stopped_since is None:
        self.stopped_since = st.t
      if mission_done and not self.finished and st.t - self.stopped_since > 0.5:
        self.finished = True
        in_zone = self.track.stop_zone is not None and Polygon(self.track.stop_zone).buffer(0.5).contains(Point(xy))
        ev.append(Event(st.t, "finish", {"in_stop_zone": bool(in_zone)}))
        self.done = True
        self.result = "finished" if in_zone else "unsafe_stop"
      elif not mission_done and st.t - self.stopped_since > RULES.standstill_timeout.value:
        ev.append(Event(st.t, "dnf", {"reason": "standstill"}))
        self.done, self.result = True, "dnf"
    else:
      self.stopped_since = None
    self.events.extend(ev)
    return ev

  # ------------------------------------------------------------------------------------------
  def lap_times(self) -> list[float]:
    d = self.track.discipline
    if d == "acceleration":
      if self.crossings and self.finish_crossings:
        return [self.finish_crossings[0] - self.crossings[0]]
      return []
    c = self.crossings
    return [b - a for a, b in zip(c[:-1], c[1:], strict=True)]

  def summary(self) -> dict:
    d = self.track.discipline
    laps = self.lap_times()
    out = {"discipline": d, "result": self.result, "laps": laps, "n_laps": len(laps), "doo": self.n_doo,
           "oc": self.n_oc, "max_speed": self.max_speed}
    doo_pen = RULES.doo_penalty_s.get(d, 2.0)
    oc_pen = RULES.oc_penalty_s.get(d, 10.0)
    dq = False
    if d == "trackdrive":
      L = self.track.length
      if len(laps) >= 3 and L * 3 / sum(laps[:3]) < RULES.dq_speed_first3.value:
        dq = True
      if any(L / t < RULES.dq_speed_other.value for t in laps[3:]):
        dq = True
      raw = sum(laps)
      out["avg_speed_first3"] = L * 3 / sum(laps[:3]) if len(laps) >= 3 else None
    elif d == "skidpad":
      raw = (laps[1] + laps[3]) / 2 if len(laps) >= 4 else None  # timed laps: 2nd right, 2nd left
    else:
      raw = sum(laps) if laps else None
    if oc_pen is None and self.n_oc:
      dq = True
    pen = self.n_doo * doo_pen + (self.n_oc * oc_pen if oc_pen else 0)
    expected = {"trackdrive": self.track.laps, "autocross": 1, "skidpad": 4, "acceleration": 1}[d]
    complete = len(laps) >= expected
    out.update({"raw_time": raw, "penalty": pen, "time": (raw + pen) if raw is not None else None, "dq": dq,
                "complete": complete, "pass": complete and not dq and self.result == "finished"})
    return out
