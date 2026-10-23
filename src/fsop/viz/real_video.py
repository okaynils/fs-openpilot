"""Run the model on a real onboard video and render the result (no ground truth available).

Frames are resampled to the model's 20 Hz, warped into the model-frame slots with the team profile,
and fed through ``fsop.runtime.FSDriver``. Layout (1280x720):
  left  : original frame with the predicted path (magenta, +-1 sigma) and detected cones (rings at their
          estimated ground position) projected back through the physical lens; below it the two
          model-frame inputs exactly as the network sees them
  right : ego top-down view (0-40 m) with detected cones and the plan; HUD with the model's own speed
          estimate and path curvature
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fsop.geometry.warp import project_to_camera
from fsop.sim.expert import T_IDXS
from fsop.track.schema import CONE_CLASSES, CONE_VIZ_RGB

PRED = (230, 60, 200)
CLS_RGB = [CONE_VIZ_RGB[c] for c in CONE_CLASSES]


def _bgr(c):
  return (int(c[2]), int(c[1]), int(c[0]))


def _plan_ground(plan: np.ndarray, std: np.ndarray | None = None, t_max: float = 6.0):
  m = T_IDXS <= t_max
  p = np.c_[plan[m, 0], plan[m, 1], np.zeros(m.sum())]
  if std is None:
    return p, None, None
  h = np.arctan2(np.gradient(plan[m, 1]), np.gradient(plan[m, 0]) + 1e-6)
  n = np.stack([-np.sin(h), np.cos(h), 0 * h], 1) * std[m, 1:2]
  return p, p - n, p + n


def _draw_poly(img, uv, ok, color, thick, scale):
  pts = (uv[ok] * scale).astype(np.int32)
  if len(pts) > 1:
    cv2.polylines(img, [pts], False, _bgr(color), thick, cv2.LINE_AA)


def _ego_map(out, size=(428, 720), rng=40.0, half_w=12.0):
  W, H = size
  img = np.full((H, W, 3), 28, np.uint8)
  s = (H - 40) / rng

  def px(p):
    return np.c_[W / 2 - p[:, 1] * s, H - 30 - p[:, 0] * s].astype(np.int32)

  for d in range(10, int(rng) + 1, 10):
    y = int(H - 30 - d * s)
    cv2.line(img, (0, y), (W, y), (55, 55, 55), 1)
    cv2.putText(img, f"{d} m", (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
  p, lo, hi = _plan_ground(out.plan, out.plan_std, t_max=10.0)
  for side in (lo, hi):
    cv2.polylines(img, [px(side)], False, _bgr(PRED), 1, cv2.LINE_AA)
  cv2.polylines(img, [px(p)], False, _bgr(PRED), 3, cv2.LINE_AA)
  for x, y, c, sc in out.cones:
    if 0 <= int(c) < 4 and abs(y) < half_w:
      cv2.circle(img, tuple(px(np.array([[x, y]]))[0]), 6, _bgr(CLS_RGB[int(c)]), -1 if sc > 0.6 else 2, cv2.LINE_AA)
  car = px(np.array([[1.6, 0.7], [1.6, -0.7], [-1.3, -0.7], [-1.3, 0.7]]))
  cv2.fillPoly(img, [car], (40, 40, 200))
  cv2.putText(img, "ego view: model cones (filled = confident) + plan", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
              (220, 220, 220), 1, cv2.LINE_AA)
  return img


def render_real_video(video: str | Path, model: str | Path, profile: str | Path, out: str | Path, start_s: float = 0.0,
                      duration_s: float | None = None, camera: str | None = None, min_score: float = 0.3,
                      label: str = "") -> dict:
  from fsop.profile import load_profile
  from fsop.runtime import FSDriver

  prof = load_profile(profile)
  cam = prof.cameras[0] if camera is None else next(c for c in prof.cameras if c.name == camera)
  drv = FSDriver(model, prof, min_score=min_score)
  T_mount = prof.mount.T_ego_cam()
  cap = cv2.VideoCapture(str(video))
  src_fps = cap.get(cv2.CAP_PROP_FPS)
  n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  end_s = n_src / src_fps if duration_s is None else min(start_s + duration_s, n_src / src_fps)
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  W, H = 1280, 720
  vw = None
  for fourcc in ("avc1", "mp4v"):
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), 20.0, (W, H))
    if vw.isOpened():
      break
  cw, ch = 852, int(round(852 * cam.height / cam.width))
  sc = cw / cam.width
  cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src_fps))
  src_i = int(start_s * src_fps)
  frame = None
  t, k, v_est, speeds = start_s, 0, 0.0, []
  while t < end_s:
    while src_i <= t * src_fps:  # nearest-previous source frame for the 20 Hz tick
      ok, frame = cap.read()
      src_i += 1
      if not ok:
        break
    if frame is None:
      break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    o = drv.step({cam.name: rgb}, v_ego=v_est)
    v_est = float(max(o.plan[0, 3], 0.0))
    speeds.append(v_est)
    canvas = np.zeros((H, W, 3), np.uint8)
    view = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_LINEAR)
    p, lo, hi = _plan_ground(o.plan, o.plan_std)
    for pts, th in ((lo, 1), (hi, 1), (p, 3)):
      uv, ok = project_to_camera(pts, T_mount, cam)
      _draw_poly(view, uv, ok, PRED, th, sc)
    if len(o.cones):
      uv, ok = project_to_camera(np.c_[o.cones[:, :2], np.zeros(len(o.cones))], T_mount, cam)
      for (u, v), f, (x, _, c, _) in zip(uv, ok, o.cones, strict=True):
        if f and 0 <= int(c) < 4:
          r = int(np.clip(90 / max(x, 1.0), 4, 22) * sc)
          cv2.circle(view, (int(u * sc), int(v * sc)), r, _bgr(CLS_RGB[int(c)]), 2, cv2.LINE_AA)
    canvas[:ch, :cw] = view
    raw = drv.warper({cam.name: rgb})
    for j, s in enumerate(("wide", "road")):
      th = cv2.resize(raw[s][..., ::-1], (426, 213), interpolation=cv2.INTER_AREA)
      cv2.putText(th, f"model input: {s}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
      y0 = min(ch, H - 213)
      canvas[y0:y0 + 213, j * 426:(j + 1) * 426] = th
    canvas[:, cw:] = _ego_map(o)
    curv = float(np.interp(1.0, T_IDXS, o.plan[:, 11]) / max(np.interp(1.0, T_IDXS, o.plan[:, 0]), 1.0))
    hud = f"t {t:5.1f}s  model speed {v_est * 3.6:4.0f} km/h  curvature {curv:+.3f} 1/m  cones {len(o.cones)}"
    cv2.rectangle(canvas, (0, 0), (cw, 26), (0, 0, 0), -1)
    cv2.putText(canvas, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
      cv2.putText(canvas, label, (8, ch - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    vw.write(canvas)
    k += 1
    t = start_s + k / 20.0
  vw.release()
  cap.release()
  return {"path": str(out), "frames": k, "mean_model_speed_kmh": float(np.mean(speeds) * 3.6) if speeds else None}
