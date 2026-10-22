"""Geometric BEV cones from the image head: bottom-centre key point -> ray -> flat-ground intersection.

The image head predicts in exact model-frame pixels (known intrinsics), so with the camera mount the
ground contact point of each cone back-projects to an ego-frame position. Detections from the road
(narrow, far range) and wide (near range) cameras are merged with per-class NMS.
"""

from __future__ import annotations

import numpy as np

from fsop.eval.det_metrics import decode_img
from fsop.geometry.camera import CAMERA_MODELS, VIEW_FROM_EGO


def ground_points(uv: np.ndarray, cam: str, T_ego_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Pixel [N,2] -> ego xy [N,2] on z=0 and a validity mask."""
  if len(uv) == 0:
    return np.zeros((0, 2)), np.zeros(0, bool)
  K = CAMERA_MODELS[cam].K
  rays_view = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
  rays_body = rays_view @ VIEW_FROM_EGO  # view -> camera body (x fwd, y left, z up)
  R, t = T_ego_cam[:3, :3], T_ego_cam[:3, 3]
  d = rays_body @ R.T
  ok = d[:, 2] < -1e-3
  s = np.where(ok, -t[2] / np.where(ok, d[:, 2], -1.0), 0.0)
  p = t[None, :2] + d[:, :2] * s[:, None]
  return p, ok & (s < 80.0)


def bev_from_image_head(img_out: dict, T_ego_cam: np.ndarray, b: int = 0, thresh: float = 0.25,
                        merge_dist: float = 0.6):
  xy_all, cls_all, sc_all = [], [], []
  for cam, max_r in (("wide", 22.0), ("road", 45.0)):
    o = img_out[cam]
    boxes, cls, sc = decode_img(o["heat"][b:b + 1], o["wh"][b:b + 1], o["off"][b:b + 1], thresh=thresh)[0]
    uv = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
    xy, ok = ground_points(uv, cam, T_ego_cam)
    ok &= np.linalg.norm(xy, axis=1) < max_r
    xy_all.append(xy[ok])
    cls_all.append(cls[ok])
    sc_all.append(sc[ok])
  xy, cls, sc = np.concatenate(xy_all), np.concatenate(cls_all), np.concatenate(sc_all)
  keep = []
  for i in np.argsort(-sc):
    if all(cls[j] != cls[i] or np.linalg.norm(xy[j] - xy[i]) > merge_dist for j in keep):
      keep.append(i)
  keep = np.array(keep, int)
  return xy[keep], cls[keep], sc[keep]
