"""Conversion between fsop's ego frame (x fwd, y left, z up) and openpilot's plan frame.

openpilot plans are expressed in the calibrated/device convention x forward, y right, z down. This was
verified empirically (M5): running the pretrained model on sim episodes, predicted plan y / yaw at
t=0.6 s correlate at -0.89 / -0.85 with ego-frame ground truth (left-positive). A 180 deg rotation
about x maps one frame to the other, so every 3-vector flips (y, z) and euler angles / body rates
flip (pitch, yaw).
"""

from __future__ import annotations

import numpy as np

PLAN_SIGN = np.array([1, -1, -1] * 5, dtype=np.float32)  # position, velocity, accel, euler, rates


def ego_to_op_plan(plan: np.ndarray) -> np.ndarray:
  """[...,33,15] ego-frame plan -> openpilot frame (self-inverse)."""
  return plan * PLAN_SIGN


op_to_ego_plan = ego_to_op_plan
