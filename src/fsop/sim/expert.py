"""Expert driver (pure pursuit + speed profile) and counterfactual plan labels.

Mission progress is tracked as unrolled arc length ``s_mission`` (lap k at k*L + s for closed loops),
so the same expert handles trackdrive (10 laps + stop), autocross, skidpad (route already unrolled in
the centerline) and acceleration (straight + stop zone).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsop.sim.vehicle import VehicleParams, VehicleState, step
from fsop.track import centerline as cl
from fsop.track.schema import Track

T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])


@dataclass
class ExpertParams:
  a_lat_max: float = 8.0
  a_acc: float = 5.0
  a_brk: float = 8.0
  v_cap: float = 14.0
  k_lookahead: float = 0.25
  lookahead_min: float = 1.8
  lookahead_max: float = 9.0
  kp_speed: float = 1.5
  stop_decel: float = 4.0
  controller: str = "pure_pursuit"  # pure_pursuit | stanley_ff
  k_lat: float = 2.0  # stanley cross-track gain
  k_head: float = 0.6
  t_preview: float = 0.12  # curvature preview [s] (actuator lag compensation)
  stop_after_line: float = 12.0  # stop this far after the final timing line (within the stop zone)


def sample_expert(rng: np.random.Generator, discipline: str) -> ExpertParams:
  p = ExpertParams(
    a_lat_max=float(rng.uniform(4.0, 11.0)),
    a_acc=float(rng.uniform(3.0, 6.5)),
    a_brk=float(rng.uniform(5.0, 10.0)),
    v_cap=float(rng.uniform(7.0, 18.0)),
    k_lookahead=float(rng.uniform(0.2, 0.35)),
  )
  if discipline == "acceleration":
    p.v_cap = float(rng.uniform(15.0, 28.0))
  if discipline == "skidpad":
    p.a_lat_max = float(rng.uniform(4.0, 9.0))
  return p


def speed_profile(kappa: np.ndarray, ds: float, closed: bool, p: ExpertParams) -> np.ndarray:
  v = np.minimum(p.v_cap, np.sqrt(p.a_lat_max / np.maximum(np.abs(kappa), 1e-4)))
  n = len(v)
  reps = 3 if closed else 1
  for _ in range(reps):  # forward (accel) / backward (brake) passes; repeated to wrap closed loops
    for i in range(1, n + (n if closed else 0)):
      a, b = (i - 1) % n, i % n
      v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * p.a_acc * ds))
    for i in range((2 * n if closed else n) - 2, -1, -1):
      a, b = (i + 1) % n, i % n
      v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * p.a_brk * ds))
  return v


class Mission:
  """Track geometry + unrolled progress bookkeeping shared by expert, referee and labels."""

  def __init__(self, track: Track, p: ExpertParams):
    self.track = track
    self.p = p
    self.pts = track.centerline
    self.closed = track.closed
    self.n = len(self.pts)
    self.s_arr = cl.arclength(self.pts, self.closed)
    self.L = track.length if self.closed else float(self.s_arr[-1])
    self.ds = self.L / self.n if self.closed else self.L / (self.n - 1)
    self.kappa = track.curvature()
    if self.closed:
      nxt = np.roll(self.pts, -1, 0)
    else:
      nxt = np.vstack([self.pts[1:], self.pts[-1:] + (self.pts[-1:] - self.pts[-2:-1])])
    self._seg = nxt - self.pts
    self._seg_l2 = np.maximum((self._seg**2).sum(1), 1e-12)
    from scipy.ndimage import gaussian_filter1d

    k_s = gaussian_filter1d(np.abs(self.kappa), 1.0 / self.ds, mode="wrap" if self.closed else "nearest")
    self.v_ref = speed_profile(np.maximum(np.abs(self.kappa), k_s), self.ds, self.closed, p)
    if self.closed:
      # staging pose is before s=0 -> mission starts at -staging; finish after `laps` crossings
      self.s_end = track.laps * self.L + p.stop_after_line
    else:
      fl = track.finish_line
      fx = 0.5 * (fl[0][0] + fl[1][0]), 0.5 * (fl[0][1] + fl[1][1])
      if track.discipline == "skidpad":
        seg = track.generator["segments"]["exit"]
        self.s_end = float(self.s_arr[seg[0]] + track.generator["R_out"] + p.stop_after_line)
      else:
        _, _, _, s_f = cl.project(self.pts, False, np.array(fx))
        self.s_end = s_f + p.stop_after_line
    self.s_end = min(self.s_end, (np.inf if self.closed else self.L - 2.0))

  # -- progress ------------------------------------------------------------------------------
  def init_progress(self, x: float, y: float) -> tuple[int, float]:
    i, _, _, s = cl.project(self.pts, self.closed, np.array([x, y]))
    if self.closed and s > self.L / 2:
      s -= self.L  # staged just before the line
    return i, s

  def update_progress(self, x: float, y: float, hint: int, s_prev: float, back: int = 30,
                      fwd: int = 120) -> tuple[int, float, float]:
    """Returns (segment index, unrolled s, lateral offset (left +)). Searches a window around ``hint``
    so self-overlapping paths (skidpad) are tracked unambiguously."""
    idx = np.arange(hint - back, hint + fwd)
    idx = idx % self.n if self.closed else np.clip(idx, 0, self.n - 2)
    a = self.pts[idx]
    ab = self._seg[idx]
    rel = np.array([x, y]) - a
    t = np.clip((rel * ab).sum(1) / self._seg_l2[idx], 0.0, 1.0)
    d = rel - ab * t[:, None]
    d2 = (d * d).sum(1)
    k = int(np.argmin(d2))
    i = int(idx[k])
    cross = ab[k, 0] * rel[k, 1] - ab[k, 1] * rel[k, 0]
    lat = float(np.sign(cross) * np.sqrt(d2[k]))
    s_local = float(self.s_arr[i] + t[k] * np.sqrt(self._seg_l2[i]))
    if self.closed:
      s = float(np.round((s_prev - s_local) / self.L) * self.L + s_local)
    else:
      s = s_local
    return i, s, lat

  def point_at(self, s: float) -> np.ndarray:
    if self.closed:
      s = s % self.L
      k = s / self.ds
      i0 = int(np.floor(k)) % self.n
      f = k - np.floor(k)
      return self.pts[i0] * (1 - f) + self.pts[(i0 + 1) % self.n] * f
    s = float(np.clip(s, 0.0, self.L))
    return np.array([np.interp(s, self.s_arr, self.pts[:, 0]), np.interp(s, self.s_arr, self.pts[:, 1])])

  def kappa_at(self, s: float) -> float:
    if self.closed:
      return float(self.kappa[int(round((s % self.L) / self.ds)) % self.n])
    return float(np.interp(s, self.s_arr, self.kappa))

  def v_at(self, s: float) -> float:
    if self.closed:
      return float(self.v_ref[int(round((s % self.L) / self.ds)) % self.n])
    return float(np.interp(s, self.s_arr, self.v_ref))

  def target_speed(self, s: float) -> float:
    v = self.v_at(s)
    rem = self.s_end - s
    v_stop = np.sqrt(max(2 * self.p.stop_decel * max(rem, 0.0), 0.0))
    return float(min(v, v_stop)) if rem > 0.3 else 0.0


class Expert:
  def __init__(self, mission: Mission, vp: VehicleParams):
    self.m = mission
    self.vp = vp

  def act(self, st: VehicleState, s: float) -> tuple[float, float]:
    p = self.m.p
    v = max(st.vx, 0.0)
    if p.controller == "pure_pursuit":
      steer = self._pure_pursuit(st, s, v)
    else:
      steer = self._stanley_ff(st, s, v)
    v_t = self.m.target_speed(s + 0.5 * v * 0.3)
    v_ahead = self.m.target_speed(s + v * 0.6)
    a_ff = (v_ahead**2 - v**2) / max(2 * max(v * 0.6, 1.0), 1e-3)
    accel = float(np.clip(p.kp_speed * (v_t - v) + 0.5 * a_ff, -p.a_brk, p.a_acc))
    if v_t <= 0.0 and v < 0.3:
      accel = -2.0
    return steer, accel


  def _pure_pursuit(self, st: VehicleState, s: float, v: float) -> float:
    p = self.m.p
    Ld = float(np.clip(p.k_lookahead * v + p.lookahead_min, p.lookahead_min, p.lookahead_max))
    rx = st.x - self.vp.lr * np.cos(st.yaw)
    ry = st.y - self.vp.lr * np.sin(st.yaw)
    tgt = self.m.point_at(s + Ld)
    dx, dy = tgt[0] - rx, tgt[1] - ry
    alpha = np.arctan2(dy, dx) - st.yaw
    return float(np.arctan2(2 * self.vp.wheelbase * np.sin(alpha), max(np.hypot(dx, dy), 1e-3)))

  def _stanley_ff(self, st: VehicleState, s: float, v: float) -> float:
    """Curvature feed-forward (previewed) + Stanley feedback at the front axle."""
    p = self.m.p
    fx = st.x + self.vp.lf * np.cos(st.yaw)
    fy = st.y + self.vp.lf * np.sin(st.yaw)
    s_f = s + self.vp.lf
    a = self.m.point_at(s_f)
    b = self.m.point_at(s_f + 0.5)
    path_h = np.arctan2(b[1] - a[1], b[0] - a[0])
    e_lat = -np.sin(path_h) * (fx - a[0]) + np.cos(path_h) * (fy - a[1])  # left of path = +
    course = st.yaw + np.arctan2(st.vy, max(st.vx, 0.5))  # velocity direction (sideslip-aware)
    e_head = np.angle(np.exp(1j * (course - path_h)))
    kappa = self.m.kappa_at(s + v * p.t_preview)
    ff = np.arctan(self.vp.wheelbase * kappa)
    return float(ff - p.k_head * e_head - np.arctan2(p.k_lat * e_lat, v + 1.0))


def rollout(mission: Mission, expert: Expert, vp: VehicleParams, st: VehicleState, s: float, hint: int,
            horizon: float = 10.0, dt: float = 0.02, ctrl_dt: float = 0.05) -> np.ndarray:
  """Noise-free closed-loop expert rollout. Returns [N,9] state rows (x,y,yaw,vx,vy,r,steer,accel,t)."""
  n = int(round(horizon / dt)) + 1
  out = np.zeros((n, 9))
  sub = int(round(ctrl_dt / dt))
  cur = st.copy()
  cur.t = 0.0
  u = expert.act(cur, s)
  for k in range(n):
    out[k] = cur.as_array()
    if k % sub == 0:
      hint, s, _ = mission.update_progress(cur.x, cur.y, hint, s)
      u = expert.act(cur, s)
    cur = step(cur, u[0], u[1], vp, dt)
  return out


def plan_from_rollout(traj: np.ndarray, dt: float = 0.02) -> np.ndarray:
  """[N,9] rollout (t0 = row 0) -> [33,15] plan in the t0 ego frame (x fwd, y left, z up).

  columns: position xyz | velocity xyz | acceleration xyz | euler (roll, pitch, yaw) rel. t0 |
           orientation rate (body roll/pitch/yaw rate)
  """
  t = traj[:, 8]
  x0, y0, yaw0 = traj[0, 0], traj[0, 1], traj[0, 2]
  c, s = np.cos(-yaw0), np.sin(-yaw0)
  px = c * (traj[:, 0] - x0) - s * (traj[:, 1] - y0)
  py = s * (traj[:, 0] - x0) + c * (traj[:, 1] - y0)
  dyaw = np.unwrap(traj[:, 2] - yaw0)
  # world-frame velocity rotated into t0 frame
  vwx = traj[:, 3] * np.cos(traj[:, 2]) - traj[:, 4] * np.sin(traj[:, 2])
  vwy = traj[:, 3] * np.sin(traj[:, 2]) + traj[:, 4] * np.cos(traj[:, 2])
  vx = c * vwx - s * vwy
  vy = s * vwx + c * vwy
  ax = np.gradient(vx, dt)
  ay = np.gradient(vy, dt)
  plan = np.zeros((33, 15))
  for j, col in enumerate((px, py, None, vx, vy, None, ax, ay, None)):
    if col is not None:
      plan[:, j] = np.interp(T_IDXS, t, col)
  plan[:, 11] = np.interp(T_IDXS, t, dyaw)
  plan[:, 14] = np.interp(T_IDXS, t, traj[:, 5])
  return plan
