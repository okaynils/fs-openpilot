"""Camera models, frames and the openpilot model-frame intrinsics.

Frames used throughout fsop:
  world   : x east, y north, z up (track coordinates)
  ego     : x forward, y left, z up, origin at vehicle reference point on the ground
  view    : x right, y down, z forward (pinhole camera / OpenCV convention)
  op calib/device: x forward, y right, z down (openpilot); conversions in ``op_frames``

Model-frame intrinsics are copied from openpilot ``common/transformations/model.py`` (MIT):
road camera input = "medmodel" (fl 910, 512x256, cy 47.6), wide = "sbigmodel" (fl 455, cy 151.8).
Rendering directly with these intrinsics (camera aligned with the calibrated frame) produces frames
that are exactly what openpilot's warp would produce for a perfectly calibrated device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODEL_W, MODEL_H = 512, 256
MEDMODEL_CY = 47.6
MEDMODEL_FL = 910.0
SBIGMODEL_FL = 455.0

MEDMODEL_K = np.array([[MEDMODEL_FL, 0.0, 0.5 * MODEL_W], [0.0, MEDMODEL_FL, MEDMODEL_CY], [0.0, 0.0, 1.0]])
SBIGMODEL_K = np.array([[SBIGMODEL_FL, 0.0, 0.5 * MODEL_W], [0.0, SBIGMODEL_FL, 0.5 * (256 + MEDMODEL_CY)],
                        [0.0, 0.0, 1.0]])

# ego (x fwd, y left, z up) -> view (x right, y down, z fwd)
VIEW_FROM_EGO = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def rot_x(a: float) -> np.ndarray:
  c, s = np.cos(a), np.sin(a)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def se3(R: np.ndarray, t) -> np.ndarray:
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = t
  return T


def se2_to_se3(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
  return se3(rot_z(yaw), [x, y, z])


@dataclass
class CameraModel:
  name: str
  K: np.ndarray
  width: int = MODEL_W
  height: int = MODEL_H

  def project_cam(self, p_cam_ego_axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points given in the camera body frame (x fwd, y left, z up). Returns uv [N,2], depth [N]."""
    v = p_cam_ego_axes @ VIEW_FROM_EGO.T
    z = v[:, 2]
    uv = (v @ self.K.T)[:, :2] / np.maximum(z[:, None], 1e-6)
    return uv, z

  def in_image(self, uv: np.ndarray, z: np.ndarray, margin: float = 0.0) -> np.ndarray:
    return (z > 0.1) & (uv[:, 0] >= -margin) & (uv[:, 0] < self.width + margin) & (uv[:, 1] >= -margin) & (
      uv[:, 1] < self.height + margin)

  @property
  def hfov(self) -> float:
    return 2 * np.arctan(self.width / 2 / self.K[0, 0])

  @property
  def vfov(self) -> float:
    return 2 * np.arctan(self.height / 2 / self.K[1, 1])


ROAD_CAM = CameraModel("road", MEDMODEL_K)
WIDE_CAM = CameraModel("wide", SBIGMODEL_K)
CAMERA_MODELS = {"road": ROAD_CAM, "wide": WIDE_CAM}


@dataclass
class CameraMount:
  """Pose of the (shared) camera origin relative to the ego frame. openpilot's prior is ~1.22 m."""

  x: float = 0.0  # forward of ego reference point
  y: float = 0.0
  z: float = 1.1
  roll: float = 0.0
  pitch: float = 0.0  # positive = nose down (looking at the ground more)
  yaw: float = 0.0

  def T_ego_cam(self) -> np.ndarray:
    R = rot_z(self.yaw) @ rot_y(self.pitch) @ rot_x(self.roll)
    return se3(R, [self.x, self.y, self.z])


def project_world(points_world: np.ndarray, T_world_ego: np.ndarray, T_ego_cam: np.ndarray, cam: CameraModel):
  """Project world points [N,3] into a camera. Returns uv [N,2], depth [N], visible mask [N]."""
  T_cam_world = np.linalg.inv(T_world_ego @ T_ego_cam)
  p = points_world @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
  uv, z = cam.project_cam(p)
  return uv, z, cam.in_image(uv, z)
