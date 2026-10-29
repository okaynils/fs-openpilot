"""Cone detection on a real, unlabelled onboard video (sim-to-real gap measure).

Pseudo ground truth comes from colour: yellow/orange cone bodies are segmented in each model-frame slot
(after the profile's warp), the black band is bridged vertically, and the blob's bottom centre is the
key point. This misses blue cones and tiny distant ones, so only cones at least ``min_h`` px tall are
scored, and detections on smaller blobs count neither way. It is a *relative* measure: compare models on
the same clip and profile, not absolute numbers against labelled datasets.

A prediction (image head key point, class-agnostic) matches a pseudo-GT cone if it lies inside the
blob's box grown by ``grow`` and at most ``max(4, h)`` px from its key point. Reports AP and P/R/F1 at
``score``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from fsop.eval.det_metrics import decode_img
from fsop.geometry.yuv import rgb_to_model

SLOTS = ("road", "wide")


def color_cones(rgb: np.ndarray) -> np.ndarray:
  """Yellow/orange cone blobs in an RGB image -> [N,6] (u_key, v_key, x1, y1, x2, y2)."""
  hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
  h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
  m = ((h >= 12) & (h <= 32) & (s >= 70) & (v >= 100)).astype(np.uint8)  # excludes green-yellow chalk
  m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 3), np.uint8))  # bridge the black band
  n, _, st, _ = cv2.connectedComponentsWithStats(m)
  boxes = [list(map(float, (x, y, x + w, y + hh))) for x, y, w, hh, a in st[1:] if a >= 3]
  boxes = _merge_stacked(boxes)  # the band can still split a cone into a top and a bottom part
  out = [((x1 + x2) / 2, y2, x1, y1, x2, y2) for x1, y1, x2, y2 in boxes
         if y2 - y1 >= 3 and 0.6 <= (y2 - y1) / max(x2 - x1, 1) <= 4.0]
  return np.array(out, np.float64).reshape(-1, 6)


def _merge_stacked(boxes: list[list[float]]) -> list[list[float]]:
  merged = True
  while merged:
    merged = False
    for i in range(len(boxes)):
      for j in range(i + 1, len(boxes)):
        a, b = boxes[i], boxes[j]
        ov = min(a[2], b[2]) - max(a[0], b[0])
        gap = max(a[1], b[1]) - min(a[3], b[3])
        u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        cone_shaped = (u[3] - u[1]) <= 2.0 * (u[2] - u[0])  # two separate cones stacked in the image are taller
        if ov >= 0.5 * min(a[2] - a[0], b[2] - b[0]) and gap <= 0.5 * min(a[3] - a[1], b[3] - b[1]) and cone_shaped:
          boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
          del boxes[j]
          merged = True
          break
      if merged:
        break
  return boxes


def horizon_v(T_ego_cam: np.ndarray, slot: str) -> float:
  """Image row of the horizon (a point far ahead on the ground) in a model-frame slot."""
  from fsop.geometry.camera import CAMERA_MODELS

  Ti = np.linalg.inv(T_ego_cam)
  pc = np.array([[1000.0, 0.0, 0.0]]) @ Ti[:3, :3].T + Ti[:3, 3]
  uv, _ = CAMERA_MODELS[slot].project_cam(pc)
  return float(uv[0, 1])


def _match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, min_h: float, grow: float):
  """Greedy by score. Returns (is_tp, counted) per prediction and the number of scored GT."""
  big = (gt[:, 5] - gt[:, 3]) >= min_h
  used = np.zeros(len(gt), bool)
  tp = np.zeros(len(pred), bool)
  counted = np.ones(len(pred), bool)
  for i in np.argsort(-scores):
    u, v = pred[i]
    best, bd = -1, np.inf
    for j, (gu, gv, x1, y1, x2, y2) in enumerate(gt):
      if used[j]:
        continue
      w, h = x2 - x1, y2 - y1
      inside = x1 - grow * w <= u <= x2 + grow * w and y1 - grow * h <= v <= y2 + grow * h
      d = np.hypot(u - gu, v - gv)
      if inside and d <= max(4.0, h) and d < bd:
        best, bd = j, d
    if best >= 0:
      used[best] = True
      tp[i] = big[best]
      counted[i] = big[best]  # a hit on a tiny (unscored) cone is neither TP nor FP
  return tp, counted, int(big.sum())


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
  if n_gt == 0 or len(scores) == 0:
    return 0.0
  o = np.argsort(-scores)
  ctp = np.cumsum(tp[o])
  prec = ctp / np.arange(1, len(o) + 1)
  rec = ctp / n_gt
  prec = np.maximum.accumulate(prec[::-1])[::-1]  # all-point interpolation
  return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


@torch.no_grad()
def evaluate_real_clip(video: str | Path, ckpt: str | Path, profile: str | Path, every_s: float = 0.25,
                       score: float = 0.3, min_h: float = 6.0, grow: float = 0.5, start_s: float = 0.0,
                       duration_s: float | None = None, debug_dir: str | Path | None = None) -> dict:
  from fsop.geometry.warp import ModelFrameWarper
  from fsop.model.build import load_fs
  from fsop.profile import load_profile
  from fsop.train.device import pick_device

  prof = load_profile(profile)
  cam = prof.cameras[0].name
  warp = ModelFrameWarper(prof)
  dev = pick_device()
  model = load_fs(ckpt).to(dev).eval()
  cap = cv2.VideoCapture(str(video))
  fps = cap.get(cv2.CAP_PROP_FPS)
  end = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
  end = end if duration_s is None else min(end, start_s + duration_s)

  def slots_at(t: float) -> dict[str, np.ndarray] | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.read()
    return warp({cam: cv2.cvtColor(f, cv2.COLOR_BGR2RGB)}) if ok else None

  acc = {s: {"scores": [], "tp": [], "n_gt": 0} for s in SLOTS}
  hz = {s: horizon_v(prof.mount.T_ego_cam(), s) for s in SLOTS}
  t = max(start_s, 0.2)
  n_frames = 0
  while t < end:
    prev, cur = slots_at(t - 0.2), slots_at(t)  # the model sees frames 0.2 s apart (t-4, t at 20 Hz)
    if prev is None or cur is None:
      break
    x = {s: torch.from_numpy(np.concatenate([rgb_to_model(prev[s]), rgb_to_model(cur[s])])[None]).to(dev).float()
         for s in SLOTS}
    v = model.base.encode(x["road"], x["wide"])
    io = model.img(v.s4, v.s8, v.s16)
    for s in SLOTS:
      boxes, _, sc = decode_img(io[s]["heat"], io[s]["wh"], io[s]["off"], thresh=0.05)[0]
      key = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]] if len(boxes) else np.zeros((0, 2))
      gt = color_cones(cur[s])
      gt = gt[gt[:, 1] > hz[s] + 3]  # cones stand on the ground: blobs at the horizon are people, banners
      tp, counted, n_gt = _match(key, sc, gt, min_h, grow)
      a = acc[s]
      a["scores"].append(sc[counted])
      a["tp"].append(tp[counted])
      a["n_gt"] += n_gt
      if debug_dir is not None and n_frames % 20 == 0:
        _debug(Path(debug_dir) / f"{s}_{t:06.2f}.png", cur[s], gt, key, sc, score, min_h)
    n_frames += 1
    t += every_s
  cap.release()
  res = {"frames": n_frames, "score_thresh": score, "min_h_px": min_h}
  for s, a in acc.items():
    sc, tp = np.concatenate(a["scores"]), np.concatenate(a["tp"])
    keep = sc >= score
    p = float(tp[keep].sum() / max(keep.sum(), 1))
    r = float(tp[keep].sum() / max(a["n_gt"], 1))
    res[s] = {"ap": average_precision(sc, tp, a["n_gt"]), "precision": p, "recall": r,
              "f1": 2 * p * r / max(p + r, 1e-9), "gt_cones": a["n_gt"], "detections": int(keep.sum())}
  return res


def _debug(path: Path, rgb, gt, key, sc, score, min_h) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  im = rgb.copy()
  for _, _, x1, y1, x2, y2 in gt:
    col = (255, 0, 255) if y2 - y1 >= min_h else (120, 120, 120)
    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, 1)
  for (u, v), s in zip(key, sc, strict=True):
    if s >= score:
      cv2.circle(im, (int(u), int(v)), 4, (0, 255, 255), 1)
  cv2.imwrite(str(path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
