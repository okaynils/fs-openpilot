"""Rerun logging: one entity schema for live sim, dataset inspection, model evaluation and replays.

Entity tree:
  world/track/{cones/<class>, centerline, left, right, timekeeping}     static
  world/ego                         Transform3D (per frame) + car box
  world/ego/plan/{gt,pred,pred_lo,pred_hi}   LineStrips3D in ego frame
  world/ego/cones_pred              Points3D (model BEV detections, ego frame)
  world/ego/cam_{road,wide}         Transform3D + Pinhole (model-frame K) + Image
  bev/{gt,pred}                     heatmap images ; bev/cones_* Points2D
  metrics/*, race/*                 Scalars / TextLog
3D entities are projected automatically into the 2D camera views because the cameras are Pinholes in
the same transform tree - this doubles as a visual check of camera conventions.
"""

from __future__ import annotations

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from fsop.geometry.camera import CAMERA_MODELS, VIEW_FROM_EGO
from fsop.track.schema import CONE_CLASSES, CONE_VIZ_RGB, ConeColor, Track

CLASS_RGB = np.array([CONE_VIZ_RGB[c] for c in CONE_CLASSES], dtype=np.uint8)


def init(app: str = "fsop", spawn: bool = False, save: str | None = None, serve: bool = False,
         blueprint: rrb.Blueprint | None = None, recording_id: str | None = None) -> None:
  rr.init(app, recording_id=recording_id, spawn=spawn)
  if save:
    rr.save(save, default_blueprint=blueprint)
  elif serve:
    rr.serve_grpc(default_blueprint=blueprint)
  if blueprint is not None and (spawn or serve):
    rr.send_blueprint(blueprint)


def log_track(track: Track, cones_xy: np.ndarray | None = None, colors: list[ConeColor] | None = None,
              path: str = "world/track") -> None:
  if cones_xy is None:
    cones_xy = np.array([[c.x, c.y] for c in track.cones])
    colors = [c.color for c in track.cones]
  for color in ConeColor:
    m = np.array([c == color for c in colors], dtype=bool)
    if m.any():
      r = 0.16 if color == ConeColor.BIG_ORANGE else 0.12
      pts = np.c_[cones_xy[m], np.full(m.sum(), 0.15)]
      rr.log(f"{path}/cones/{color.value}", rr.Points3D(pts, radii=r, colors=CONE_VIZ_RGB[color]), static=True)
  left, right = track.boundaries()
  close = (lambda a: np.vstack([a, a[:1]])) if track.closed else (lambda a: a)
  z = lambda a: np.c_[a, np.zeros(len(a))]  # noqa: E731
  rr.log(f"{path}/centerline", rr.LineStrips3D([z(close(track.centerline))], colors=(150, 150, 150), radii=0.03),
         static=True)
  rr.log(f"{path}/bounds", rr.LineStrips3D([z(close(left)), z(close(right))], colors=[(60, 100, 230), (240, 200, 30)],
                                           radii=0.02), static=True)
  if track.timekeeping_line:
    a, b = track.timekeeping_line
    rr.log(f"{path}/timekeeping", rr.LineStrips3D([[[*a, 0.01], [*b, 0.01]]], colors=(255, 255, 255), radii=0.05),
           static=True)


def log_cameras(T_ego_cam: np.ndarray, cams: tuple[str, ...] = ("road", "wide"), path: str = "world/ego") -> None:
  """Camera entities: transform (ego <- view frame) + pinhole with the model-frame intrinsics."""
  R = T_ego_cam[:3, :3] @ VIEW_FROM_EGO.T
  for name in cams:
    cam = CAMERA_MODELS[name]
    rr.log(f"{path}/cam_{name}", rr.Transform3D(translation=T_ego_cam[:3, 3], mat3x3=R))
    rr.log(f"{path}/cam_{name}", rr.Pinhole(image_from_camera=cam.K, resolution=[cam.width, cam.height],
                                            camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=1.0))


def log_ego(T_world_ego: np.ndarray, vp=None, path: str = "world/ego") -> None:
  rr.log(path, rr.Transform3D(translation=T_world_ego[:3, 3], mat3x3=T_world_ego[:3, :3]))
  if vp is not None:
    f, r = vp.lf + vp.front_overhang, -(vp.lr + vp.rear_overhang)
    rr.log(f"{path}/body", rr.Boxes3D(centers=[[(f + r) / 2, 0, 0.35]], half_sizes=[[(f - r) / 2, vp.width / 2, 0.35]],
                                      colors=(200, 30, 30), fill_mode="solid"), static=True)


def log_images(images: dict[str, np.ndarray], path: str = "world/ego", jpeg: int | None = 85) -> None:
  for name, img in images.items():
    im = rr.Image(img)
    rr.log(f"{path}/cam_{name}/image", im.compress(jpeg_quality=jpeg) if jpeg else im)


def plan_strip(plan: np.ndarray, z: float = 0.05) -> np.ndarray:
  """[33,15] plan (ego frame) -> [33,3] polyline."""
  return np.c_[plan[:, 0], plan[:, 1], np.full(len(plan), z)]


def log_plan(name: str, plan: np.ndarray, color, std: np.ndarray | None = None, path: str = "world/ego/plan",
             radius: float = 0.06) -> None:
  rr.log(f"{path}/{name}", rr.LineStrips3D([plan_strip(plan)], colors=color, radii=radius))
  if std is not None:
    h = np.arctan2(np.gradient(plan[:, 1]), np.gradient(plan[:, 0]) + 1e-6)
    n = np.stack([-np.sin(h), np.cos(h)], 1) * std[:, 1:2]
    lo, hi = plan[:, :2] - n, plan[:, :2] + n
    rr.log(f"{path}/{name}_band", rr.LineStrips3D([np.c_[lo, np.zeros(33)], np.c_[hi, np.zeros(33)]],
                                                  colors=(*color[:3], 110), radii=0.025))


def log_cones_ego(name: str, xy: np.ndarray, cls: np.ndarray, score: np.ndarray | None = None,
                  path: str = "world/ego") -> None:
  if len(xy) == 0:
    rr.log(f"{path}/{name}", rr.Clear(recursive=False))
    return
  cols = CLASS_RGB[np.clip(cls, 0, len(CLASS_RGB) - 1)]
  if score is not None:
    cols = np.c_[cols, (np.clip(score, 0.2, 1.0) * 255).astype(np.uint8)]
  rr.log(f"{path}/{name}", rr.Points3D(np.c_[xy, np.full(len(xy), 0.35)], colors=cols, radii=0.09))


def log_scalars(prefix: str, values: dict[str, float]) -> None:
  for k, v in values.items():
    if v is not None and np.isfinite(v):
      rr.log(f"{prefix}/{k}", rr.Scalars(float(v)))


def log_events(events, path: str = "race/events") -> None:
  for e in events:
    level = {"doo": "WARN", "off_course": "ERROR", "dnf": "ERROR", "finish": "INFO"}.get(e.kind, "INFO")
    rr.log(path, rr.TextLog(f"{e.kind} {e.data}", level=level))


def bev_image(heat: np.ndarray) -> np.ndarray:
  """[C,H,W] class heatmaps (x forward = up, y left = left) -> RGB image."""
  C = heat.shape[0]
  rgb = np.zeros((*heat.shape[1:], 3), np.float32)
  for c in range(C):
    rgb = np.maximum(rgb, heat[c][..., None] * CLASS_RGB[c][None, None, :].astype(np.float32))
  img = rgb[::-1]  # rows: far x at the top; columns already run from +y (left) to -y (right)
  return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
def sim_blueprint(cams: tuple[str, ...] = ("road", "wide"), bev: bool = False) -> rrb.Blueprint:
  cam_views = [rrb.Spatial2DView(origin=f"world/ego/cam_{c}", name=f"{c} camera",
                                 contents=["$origin/**", "world/track/cones/**", "world/ego/plan/**",
                                           "world/ego/cones_pred"]) for c in cams]
  right = cam_views + ([rrb.Spatial2DView(origin="bev", name="BEV cones")] if bev else [])
  return rrb.Blueprint(
    rrb.Horizontal(
      rrb.Vertical(
        rrb.Spatial3DView(origin="world", name="world"),
        rrb.TimeSeriesView(origin="metrics", name="metrics"),
        row_shares=[3, 1],
      ),
      rrb.Vertical(*right, rrb.TextLogView(origin="race", name="race events")),
      column_shares=[3, 2],
    ),
    collapse_panels=True,
  )
