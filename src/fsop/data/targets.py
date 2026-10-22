"""Training targets derived on the fly from episode cones + ego pose + camera mount.

BEV grid (ego frame, x forward, y left): x in [X0, X1), y in [-Y, Y) with ``cell`` metres per cell.
  heat   [C, H, W]  CenterNet gaussians per cone class (row = x index, col = y index from +Y to -Y)
  offset [2, H, W]  sub-cell offsets (dx, dy) in cells at positive cells
  mask   [H, W]     1 where the cell is observable (inside the wide-camera frustum, in range)
Image targets (per camera, stride ``IMG_STRIDE`` on the 512x256 model frame):
  heat [C, h, w], wh [2, h, w] (box size / stride), off [2, h, w], mask (positive cells)
  key point = bottom centre of the cone (ground contact), consistent with the BEV target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fsop.geometry.camera import CAMERA_MODELS, VIEW_FROM_EGO
from fsop.track.rules import RULES

N_CLS = 4


@dataclass(frozen=True)
class BevGrid:
  x0: float = 0.0
  x1: float = 48.0
  y: float = 24.0
  cell: float = 0.75
  sigma_m: float = 0.5
  max_range: float = 45.0
  min_x: float = 2.0

  @property
  def H(self) -> int:
    return int(round((self.x1 - self.x0) / self.cell))

  @property
  def W(self) -> int:
    return int(round(2 * self.y / self.cell))

  def to_cell(self, xy: np.ndarray) -> np.ndarray:
    """ego xy [N,2] -> continuous (row, col)."""
    r = (xy[:, 0] - self.x0) / self.cell
    c = (self.y - xy[:, 1]) / self.cell
    return np.stack([r, c], 1)

  def from_cell(self, rc: np.ndarray) -> np.ndarray:
    x = rc[:, 0] * self.cell + self.x0
    y = self.y - rc[:, 1] * self.cell
    return np.stack([x, y], 1)

  def cell_centers(self) -> np.ndarray:
    r, c = np.meshgrid(np.arange(self.H) + 0.5, np.arange(self.W) + 0.5, indexing="ij")
    return self.from_cell(np.stack([r.ravel(), c.ravel()], 1)).reshape(self.H, self.W, 2)


GRID = BevGrid()
IMG_STRIDE = 4


def world_to_ego(xy_world: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
  c, s = np.cos(-yaw), np.sin(-yaw)
  d = xy_world - np.array([x, y])
  return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1)


def frustum_mask(grid: BevGrid, T_ego_cam: np.ndarray, cam: str = "wide", margin: float = 0.03) -> np.ndarray:
  """Cells whose ground point projects inside the camera image (with a tolerance) and are in range."""
  pts = grid.cell_centers().reshape(-1, 2)
  p = np.c_[pts, np.zeros(len(pts))]
  T = np.linalg.inv(T_ego_cam)
  pc = p @ T[:3, :3].T + T[:3, 3]
  m = CAMERA_MODELS[cam]
  uv, z = m.project_cam(pc)
  w, h = m.width, m.height
  ok = (z > 0.5) & (uv[:, 0] > -margin * w) & (uv[:, 0] < (1 + margin) * w) & (uv[:, 1] < h * 1.15)
  rng = np.linalg.norm(pts, axis=1)
  ok &= (rng < grid.max_range) & (pts[:, 0] > grid.min_x)
  return ok.reshape(grid.H, grid.W).astype(np.float32)


def visible_cones(cones_ego: np.ndarray, T_ego_cam: np.ndarray, cams=("road", "wide")) -> np.ndarray:
  """Cone is visible if its base or top projects into any camera."""
  vis = np.zeros(len(cones_ego), bool)
  if len(cones_ego) == 0:
    return vis
  T = np.linalg.inv(T_ego_cam)
  for zc in (0.02, RULES.small_cone.height):
    p = np.c_[cones_ego, np.full(len(cones_ego), zc)]
    pc = p @ T[:3, :3].T + T[:3, 3]
    for name in cams:
      uv, z = CAMERA_MODELS[name].project_cam(pc)
      vis |= CAMERA_MODELS[name].in_image(uv, z)
  return vis


def bev_targets(cones_ego: np.ndarray, cls: np.ndarray, mask: np.ndarray, grid: BevGrid = GRID):
  heat = np.zeros((N_CLS, grid.H, grid.W), np.float32)
  off = np.zeros((2, grid.H, grid.W), np.float32)
  pos = np.zeros((grid.H, grid.W), np.float32)
  sig = grid.sigma_m / grid.cell
  rad = int(np.ceil(3 * sig))
  rc = grid.to_cell(cones_ego)
  for (r, c), k in zip(rc, cls, strict=True):
    if k < 0 or k >= N_CLS:
      continue
    ri, ci = int(np.floor(r)), int(np.floor(c))
    if not (0 <= ri < grid.H and 0 <= ci < grid.W):
      continue
    r0, r1 = max(ri - rad, 0), min(ri + rad + 1, grid.H)
    c0, c1 = max(ci - rad, 0), min(ci + rad + 1, grid.W)
    rr, cc = np.meshgrid(np.arange(r0, r1) + 0.5, np.arange(c0, c1) + 0.5, indexing="ij")
    g = np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sig**2))
    heat[k, r0:r1, c0:c1] = np.maximum(heat[k, r0:r1, c0:c1], g)
    heat[k, ri, ci] = 1.0
    off[:, ri, ci] = (r - ri, c - ci)
    pos[ri, ci] = 1.0
  return heat, off, pos, mask


def cone_boxes_image(cones_ego: np.ndarray, cls_big: np.ndarray, T_ego_cam: np.ndarray, cam: str):
  """Project cones into a camera. Returns key points (bottom centre) [N,2], boxes [N,4] xyxy, valid [N]."""
  m = CAMERA_MODELS[cam]
  T = np.linalg.inv(T_ego_cam)
  n = len(cones_ego)
  if n == 0:
    return np.zeros((0, 2)), np.zeros((0, 4)), np.zeros(0, bool)
  # sample the cone silhouette: base square corners + top point
  pts = []
  for big in (False, True):
    spec = RULES.large_cone if big else RULES.small_cone
    b = spec.base / 2
    pts.append(np.array([[-b, -b, 0], [b, -b, 0], [b, b, 0], [-b, b, 0], [0, 0, spec.height], [0, 0, 0]]))
  key = np.zeros((n, 2))
  box = np.zeros((n, 4))
  valid = np.zeros(n, bool)
  for i in range(n):
    P = pts[int(cls_big[i])] + np.r_[cones_ego[i], 0.0]
    pc = P @ T[:3, :3].T + T[:3, 3]
    uv, z = m.project_cam(pc)
    if (z < 0.3).any():
      continue
    key[i] = uv[-1]
    box[i] = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
    valid[i] = (box[i, 2] > 0) & (box[i, 0] < m.width) & (box[i, 3] > 0) & (box[i, 1] < m.height) & (
      0 <= key[i, 0] < m.width) & (0 <= key[i, 1] < m.height * 1.2)
  return key, box, valid


def image_targets(key: np.ndarray, box: np.ndarray, cls: np.ndarray, valid: np.ndarray, W: int = 512, H: int = 256,
                  stride: int = IMG_STRIDE, min_size: float = 2.0):
  h, w = H // stride, W // stride
  heat = np.zeros((N_CLS, h, w), np.float32)
  wh = np.zeros((2, h, w), np.float32)
  off = np.zeros((2, h, w), np.float32)
  pos = np.zeros((h, w), np.float32)
  for (u, v), bx, k, ok in zip(key, box, cls, valid, strict=True):
    if not ok or k < 0 or k >= N_CLS:
      continue
    bw, bh = (bx[2] - bx[0]) / stride, (bx[3] - bx[1]) / stride
    if max(bw, bh) * stride < min_size:
      continue
    cu, cv = u / stride, min(v / stride, h - 1e-3)
    ci, ri = int(cu), int(cv)
    if not (0 <= ri < h and 0 <= ci < w):
      continue
    sig = max(0.6, 0.15 * np.hypot(bw, bh))
    rad = int(np.ceil(3 * sig))
    r0, r1 = max(ri - rad, 0), min(ri + rad + 1, h)
    c0, c1 = max(ci - rad, 0), min(ci + rad + 1, w)
    rr, cc = np.meshgrid(np.arange(r0, r1) + 0.5, np.arange(c0, c1) + 0.5, indexing="ij")
    g = np.exp(-((rr - cv) ** 2 + (cc - cu) ** 2) / (2 * sig**2))
    heat[k, r0:r1, c0:c1] = np.maximum(heat[k, r0:r1, c0:c1], g)
    heat[k, ri, ci] = 1.0
    wh[:, ri, ci] = (bw, bh)
    off[:, ri, ci] = (cu - ci, cv - ri)
    pos[ri, ci] = 1.0
  return heat, wh, off, pos


_ = VIEW_FROM_EGO
