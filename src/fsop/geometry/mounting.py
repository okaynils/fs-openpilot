"""Episode-fixed installation, calibration error, and smooth deterministic mount flex.

All translations are metres relative to the ground-projected vehicle CoG. Angles
are radians, with the CameraMount Rz(yaw) Ry(pitch) Rx(roll) convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fsop.geometry.camera import CameraMount


@dataclass
class MountDynamics:
  translation_amplitude: float = 0.0
  angle_amplitude_deg: float = 0.0
  frequency_hz: float = 1.0
  phase: list[float] = field(default_factory=lambda: [0.0] * 6)

  def __post_init__(self):
    values = [self.translation_amplitude, self.angle_amplitude_deg, self.frequency_hz, *self.phase]
    if len(self.phase) != 6 or not np.isfinite(values).all():
      raise ValueError("Mount motion requires six finite phases and finite parameters")
    if self.translation_amplitude < 0 or self.angle_amplitude_deg < 0 or self.frequency_hz <= 0:
      raise ValueError("Mount motion amplitudes must be nonnegative and frequency positive")

  def pose(self, installation: CameraMount, time: float) -> CameraMount:
    # Analytic motion: independent of rendering cadence and repeated observe() calls.
    a = np.array([self.translation_amplitude] * 3 + [np.deg2rad(self.angle_amplitude_deg)] * 3)
    frequencies = self.frequency_hz * np.array([0.71, 0.83, 1.0, 0.79, 0.93, 0.67])
    delta = a * np.sin(2 * np.pi * frequencies * time + self.phase)
    names = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
    return CameraMount(**{k: getattr(installation, k) + float(v) for k, v in zip(names, delta, strict=True)})


def ray_box_distance(origin: np.ndarray, directions: np.ndarray, bounds: np.ndarray) -> np.ndarray:
  """First positive ray/AABB distance, infinity for misses; direction need not be normalized."""
  directions = np.asarray(directions)
  parallel = np.abs(directions) < 1e-12
  safe = np.where(parallel, 1.0, directions)
  a, b = (bounds[0] - origin) / safe, (bounds[1] - origin) / safe
  lo, hi = np.minimum(a, b), np.maximum(a, b)
  inside = (origin >= bounds[0]) & (origin <= bounds[1])
  lo = np.where(parallel, np.where(inside, -np.inf, np.inf), lo)
  hi = np.where(parallel, np.where(inside, np.inf, -np.inf), hi)
  enter, leave = lo.max(-1), hi.min(-1)
  return np.where((leave >= np.maximum(enter, 0)) & (leave > 0), np.maximum(enter, 0), np.inf)


def body_bounds(vehicle) -> np.ndarray:
  # A deliberately simple low chassis box, not a claim of photorealistic bodywork.
  return np.array([[-vehicle.lr - vehicle.rear_overhang, -vehicle.width / 2, 0.0],
                   [vehicle.lf + vehicle.front_overhang, vehicle.width / 2, vehicle.body_height]])


def body_occluded(points: np.ndarray, transform: np.ndarray, vehicle) -> np.ndarray:
  if vehicle.body_height <= 0:
    return np.zeros(len(points), bool)
  origin = transform[:3, 3]
  return ray_box_distance(origin, np.asarray(points) - origin, body_bounds(vehicle)) < 1.0


def body_image_mask(camera, transform: np.ndarray, vehicle) -> np.ndarray:
  from fsop.geometry.camera import VIEW_FROM_EGO

  u, v = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
  rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(camera.K).T
  directions = rays @ VIEW_FROM_EGO @ transform[:3, :3].T
  return np.isfinite(ray_box_distance(transform[:3, 3], directions, body_bounds(vehicle)))
