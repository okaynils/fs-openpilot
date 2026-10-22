"""Warp real camera images into openpilot's model-frame input slots ("road" = medmodel, "wide" = sbigmodel).

The virtual model-frame camera sits at the profile's nominal ``mount`` orientation - exactly the camera
the simulator renders, so real and simulated inputs share geometry. A physical camera's own
``roll/pitch/yaw_deg`` (``CameraSpec``) are its deviation from that nominal mount (e.g. from extrinsic
calibration). Lens distortion (OpenCV radtan or fisheye) is undone in the same remap. One physical
camera can feed both slots: the wide slot then shows its full FOV, the road slot a 2x zoomed centre.

This replaces openpilot's calibration-based ``get_warp_matrix`` for FS cars (fixed mount, known
calibration) and runs on RGB with bilinear sampling before ``rgb_to_model``.
"""

from __future__ import annotations

import cv2
import numpy as np

from fsop.geometry.camera import CAMERA_MODELS, VIEW_FROM_EGO, rot_x, rot_y, rot_z


def _cam_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
  return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def slot_maps(cam, slot: str) -> tuple[np.ndarray, np.ndarray]:
  """Remap tables (model-slot pixel -> source pixel) for ``cv2.remap``."""
  m = CAMERA_MODELS[slot]
  u, v = np.meshgrid(np.arange(m.width, dtype=np.float64), np.arange(m.height, dtype=np.float64))
  rays = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], 1) @ np.linalg.inv(m.K).T  # virtual view frame
  body = rays @ VIEW_FROM_EGO  # -> virtual camera body (x fwd, y left, z up)
  R = _cam_rot(*np.deg2rad([cam.roll_deg, cam.pitch_deg, cam.yaw_deg]))
  phys = body @ R  # R^T applied row-wise: into the physical camera body
  view = phys @ VIEW_FROM_EGO.T
  z = view[:, 2]
  behind = z <= 1e-6
  view = view / np.where(behind, 1.0, z)[:, None]
  pts = view[:, :2].reshape(-1, 1, 2)
  if cam.fisheye:
    d = np.zeros(4) if not cam.dist else np.asarray(cam.dist[:4], np.float64)
    src = cv2.fisheye.distortPoints(pts, cam.K, d)
  else:
    d = np.zeros(5) if not cam.dist else np.asarray(cam.dist, np.float64)
    src, _ = cv2.projectPoints(np.c_[pts.reshape(-1, 2), np.ones(len(z))], np.zeros(3), np.zeros(3), cam.K, d)
  src = src.reshape(-1, 2)
  src[behind] = -1e4
  return (src[:, 0].reshape(m.height, m.width).astype(np.float32),
          src[:, 1].reshape(m.height, m.width).astype(np.float32))


class ModelFrameWarper:
  """Builds remap tables once per (camera, slot) of a TeamProfile; ``__call__`` maps raw images -> slots."""

  def __init__(self, profile, slots: tuple[str, ...] = ("road", "wide")):
    self.cams = {s: profile.camera_for(s) for s in slots}
    self.maps = {s: slot_maps(c, s) for s, c in self.cams.items()}

  def __call__(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """images: physical camera name -> RGB uint8 -> {slot: 256x512 RGB}"""
    out = {}
    for slot, cam in self.cams.items():
      mx, my = self.maps[slot]
      out[slot] = cv2.remap(images[cam.name], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    return out
