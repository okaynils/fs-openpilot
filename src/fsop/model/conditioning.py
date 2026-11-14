"""Versioned calibrated geometry contract, shared by dataset and deployment.

Vision history remains unconditioned, so frozen vision caches are reusable between
image-only and conditioned ablations. No simulator state enters this vector.
"""
from __future__ import annotations

import numpy as np

CONTEXT_VERSION = 1
CONTEXT_FIELDS = ("camera_x", "camera_y", "camera_z", "camera_roll", "camera_pitch", "camera_yaw",
                  "wheelbase", "width", "front_extent", "rear_extent", "max_steer", "max_speed",
                  "max_accel", "max_brake", "tau_steer", "tau_accel")
CONTEXT_SCALE = np.array([1, 1, 1, .1, .1, .1, 2, 2, 2, 2, .5, 15, 7, 13, .2, .2], np.float32)
CONTEXT_V2_INDICES = (0, 1, 2, *range(6, 16))


def context_indices(version=1):
  if version == 1:
    return tuple(range(16))
  if version == 2:
    return CONTEXT_V2_INDICES
  raise ValueError(f"Unsupported geometry context version: {version}")


def context_fields(version=1):
  return tuple(CONTEXT_FIELDS[i] for i in context_indices(version))


def context_scale(version=1):
  return CONTEXT_SCALE[list(context_indices(version))]


def prune_orientation_state(state):
  """Exactly specialize a v1 anchored adapter to zero angle inputs; retain source tensors."""
  if state['context_adapter.0.weight'].shape[1] != 528 or bool(state['context_anchor'][3:6].any()):
    raise ValueError('Orientation pruning requires a v1 adapter anchored at zero camera angles')
  result = {key: value.clone() for key, value in state.items()}
  columns = [*range(512), *(512 + i for i in CONTEXT_V2_INDICES)]
  result['context_adapter.0.weight'] = result['context_adapter.0.weight'][:, columns].clone()
  result['context_anchor'] = result['context_anchor'][list(CONTEXT_V2_INDICES)].clone()
  return result


def geometry_context(mount, vehicle, version=1) -> np.ndarray:
  v = vehicle
  values = [mount.x, mount.y, mount.z, mount.roll, mount.pitch, mount.yaw,
            v.wheelbase, v.width, v.lf + v.front_overhang, v.lr + v.rear_overhang,
            v.max_steer, v.max_speed, v.max_accel, v.max_brake, v.tau_steer, v.tau_accel]
  x = np.asarray(values, np.float32)
  if not np.isfinite(x).all() or mount.z <= 0 or np.any(x[6:] <= 0):
    raise ValueError("Geometry context requires finite calibration and positive vehicle limits")
  return (x / CONTEXT_SCALE)[list(context_indices(version))]
