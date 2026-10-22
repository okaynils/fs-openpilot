"""FS-car vehicle model: kinematic bicycle at low speed blended into a dynamic bicycle with saturating
linear tires, first-order actuator lags. Reference point = centre of gravity, on the ground."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VehicleParams:
  mass: float = 230.0
  iz: float = 110.0
  lf: float = 0.80
  lr: float = 0.75
  track: float = 1.20
  length: float = 2.90
  width: float = 1.40
  front_overhang: float = 0.80  # from front axle to nose (CoG -> nose = lf + this)
  rear_overhang: float = 0.55
  cf: float = 32000.0  # cornering stiffness front [N/rad] (axle)
  cr: float = 36000.0
  mu: float = 1.4
  max_steer: float = np.deg2rad(24.0)
  max_steer_rate: float = np.deg2rad(150.0)
  max_accel: float = 7.0
  max_brake: float = 13.0
  tau_steer: float = 0.06
  tau_accel: float = 0.12
  drag: float = 0.6  # 0.5 * rho * Cd * A [kg/m]
  v_blend: tuple[float, float] = (2.0, 5.0)  # kinematic -> dynamic blend range

  @property
  def wheelbase(self) -> float:
    return self.lf + self.lr

  def footprint(self) -> np.ndarray:
    """Body rectangle corners in ego frame [4,2]."""
    f, r, w = self.lf + self.front_overhang, -(self.lr + self.rear_overhang), self.width / 2
    return np.array([[f, w], [f, -w], [r, -w], [r, w]])

  def wheels(self) -> np.ndarray:
    t = self.track / 2
    return np.array([[self.lf, t], [self.lf, -t], [-self.lr, -t], [-self.lr, t]])


@dataclass
class VehicleState:
  x: float = 0.0
  y: float = 0.0
  yaw: float = 0.0
  vx: float = 0.0  # body-frame longitudinal speed
  vy: float = 0.0
  r: float = 0.0  # yaw rate
  steer: float = 0.0  # actual front wheel angle
  accel: float = 0.0  # actual longitudinal acceleration command after lag
  t: float = 0.0
  extra: dict = field(default_factory=dict)

  def copy(self) -> VehicleState:
    return VehicleState(self.x, self.y, self.yaw, self.vx, self.vy, self.r, self.steer, self.accel, self.t)

  @property
  def speed(self) -> float:
    return float(np.hypot(self.vx, self.vy))

  def as_array(self) -> np.ndarray:
    return np.array([self.x, self.y, self.yaw, self.vx, self.vy, self.r, self.steer, self.accel, self.t])


def step(s: VehicleState, steer_cmd: float, accel_cmd: float, p: VehicleParams, dt: float) -> VehicleState:
  """One integration step (semi-implicit Euler)."""
  steer_cmd = float(np.clip(steer_cmd, -p.max_steer, p.max_steer))
  accel_cmd = float(np.clip(accel_cmd, -p.max_brake, p.max_accel))
  d_steer = (steer_cmd - s.steer) * min(dt / p.tau_steer, 1.0)
  d_steer = float(np.clip(d_steer, -p.max_steer_rate * dt, p.max_steer_rate * dt))
  steer = s.steer + d_steer
  acc = s.accel + (accel_cmd - s.accel) * min(dt / p.tau_accel, 1.0)
  v = max(s.vx, 0.0)
  # friction-circle limit on longitudinal accel
  acc = float(np.clip(acc, -p.mu * 9.81, p.mu * 9.81 * 0.8))
  a_long = acc - p.drag * v * v / p.mass
  L = p.wheelbase

  # kinematic bicycle (CoG)
  beta = np.arctan(p.lr / L * np.tan(steer))
  vx_k = max(v + a_long * dt, 0.0)
  r_k = vx_k / p.lr * np.sin(beta)
  vy_k = vx_k * np.tan(beta)

  # dynamic bicycle
  vx_d, vy_d, r_d = v, s.vy, s.r
  if v > 0.5:
    af = np.arctan2(s.vy + p.lf * s.r, v) - steer
    ar = np.arctan2(s.vy - p.lr * s.r, v)
    fz_f = p.mass * 9.81 * p.lr / L
    fz_r = p.mass * 9.81 * p.lf / L
    fyf = float(np.clip(-p.cf * af, -p.mu * fz_f, p.mu * fz_f))
    fyr = float(np.clip(-p.cr * ar, -p.mu * fz_r, p.mu * fz_r))
    vx_d = max(v + (a_long - fyf * np.sin(steer) / p.mass + s.r * s.vy) * dt, 0.0)
    vy_d = s.vy + ((fyf * np.cos(steer) + fyr) / p.mass - s.r * v) * dt
    r_d = s.r + ((p.lf * fyf * np.cos(steer) - p.lr * fyr) / p.iz) * dt

  w = float(np.clip((v - p.v_blend[0]) / (p.v_blend[1] - p.v_blend[0]), 0.0, 1.0))
  vx = (1 - w) * vx_k + w * vx_d
  vy = (1 - w) * vy_k + w * vy_d
  r = (1 - w) * r_k + w * r_d
  if vx < 1e-3:
    vx, vy, r = 0.0, 0.0, 0.0
  yaw = s.yaw + r * dt
  c, sn = np.cos(yaw), np.sin(yaw)
  x = s.x + (vx * c - vy * sn) * dt
  y = s.y + (vx * sn + vy * c) * dt
  return VehicleState(float(x), float(y), float(yaw), float(vx), float(vy), float(r), float(steer), float(acc),
                      s.t + dt)


def body_accel(prev: VehicleState, cur: VehicleState, dt: float) -> tuple[float, float]:
  """Body-frame acceleration estimate (for labels)."""
  ax = (cur.vx - prev.vx) / dt - cur.r * cur.vy
  ay = (cur.vy - prev.vy) / dt + cur.r * cur.vx
  return ax, ay
