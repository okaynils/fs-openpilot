"""Plan followers: turn a predicted plan (ego frame) into steering / acceleration commands."""

from __future__ import annotations

import numpy as np

from fsop.sim.expert import T_IDXS
from fsop.sim.vehicle import VehicleParams

MIN_STABLE_DELAY = 0.05  # openpilot drive_helpers


class PlanPurePursuit:
  """Pure pursuit on the predicted path (lookahead point at a time horizon) + speed from the plan."""

  def __init__(self, vp: VehicleParams, t_min: float = 0.35, t_max: float = 1.0, k: float = 0.03, kp_v: float = 1.5,
               action_t: float = 0.3):
    self.vp, self.t_min, self.t_max, self.k, self.kp_v, self.action_t = vp, t_min, t_max, k, kp_v, action_t

  def __call__(self, plan: np.ndarray, v: float) -> tuple[float, float]:
    t_la = float(np.clip(self.t_min + self.k * v, self.t_min, self.t_max))
    x = np.interp(t_la, T_IDXS, plan[:, 0])
    y = np.interp(t_la, T_IDXS, plan[:, 1])
    # path is relative to the CoG; pure pursuit geometry from the rear axle
    xr = x + self.vp.lr
    ld2 = max(xr * xr + y * y, (self.vp.wheelbase / 1.55) ** 2)
    steer = float(np.arctan(2 * self.vp.wheelbase * y / ld2))
    v_tgt = float(np.clip(np.interp(self.action_t, T_IDXS, plan[:, 3]), 0, self.vp.max_speed))
    a_plan = float(np.interp(self.action_t, T_IDXS, plan[:, 6]))
    accel = self.kp_v * (v_tgt - v) + 0.5 * a_plan
    return (float(np.clip(steer, -self.vp.max_steer, self.vp.max_steer)),
            float(np.clip(accel, -self.vp.max_brake, self.vp.max_accel)))


class OpenpilotFollower:
  """Port of openpilot's get_curvature_from_plan / get_accel_from_plan (MIT), with a kinematic steer map."""

  def __init__(self, vp: VehicleParams, lat_action_t: float = 0.2, long_action_t: float = 0.3):
    self.vp, self.lat_t, self.long_t = vp, lat_action_t, long_action_t

  def __call__(self, plan: np.ndarray, v: float) -> tuple[float, float]:
    yaws, yaw_rates = plan[:, 11], plan[:, 14]
    vego = max(v, 1.0)
    if self.lat_t < MIN_STABLE_DELAY:
      psi = (self.lat_t / MIN_STABLE_DELAY) * np.interp(MIN_STABLE_DELAY, T_IDXS, yaws)
    else:
      psi = np.interp(self.lat_t, T_IDXS, yaws)
    curv = 2 * psi / (vego * self.lat_t) - yaw_rates[0] / vego
    steer = float(np.arctan(self.vp.wheelbase * curv))
    speeds, accels = plan[:, 3], plan[:, 6]
    v_target = np.clip(np.interp(self.long_t, T_IDXS, speeds), 0, self.vp.max_speed)
    a_target = 2 * (v_target - speeds[0]) / self.long_t - accels[0]
    if v > self.vp.max_speed:
      a_target = min(a_target, -1.5 * (v - self.vp.max_speed))
    return (float(np.clip(steer, -self.vp.max_steer, self.vp.max_steer)),
            float(np.clip(a_target, -self.vp.max_brake, self.vp.max_accel)))


def make_controller(name: str, vp: VehicleParams):
  return {"pp": PlanPurePursuit, "op": OpenpilotFollower}[name](vp)
