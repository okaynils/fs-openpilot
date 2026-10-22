"""Image-space cone detection metrics (COCO mAP via pycocotools) for the image head and YOLO.

Boxes are reconstructed from CenterNet outputs: key point = bottom centre, (w, h) regressed:
  x1 = u - w/2, x2 = u + w/2, y1 = v - h, y2 = v
The same reconstruction is applied to GT targets, so both share one definition.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import torch

from fsop.data.targets import IMG_STRIDE, N_CLS


def decode_img(heat: torch.Tensor, wh: torch.Tensor, off: torch.Tensor, thresh: float = 0.1, k: int = 100):
  """[B,C,h,w] -> per batch (boxes [N,4] xyxy in 512x256 px, cls [N], score [N])."""
  p = torch.sigmoid(heat.float())
  peak = (p == torch.nn.functional.max_pool2d(p, 3, 1, 1)) & (p > thresh)
  o = torch.sigmoid(off.float())
  out = []
  for b in range(p.shape[0]):
    c, r, cc = torch.nonzero(peak[b], as_tuple=True)
    s = p[b, c, r, cc]
    if len(s) > k:
      t = torch.topk(s, k).indices
      c, r, cc, s = c[t], r[t], cc[t], s[t]
    u = (cc.float() + o[b, 0, r, cc]) * IMG_STRIDE
    v = (r.float() + o[b, 1, r, cc]) * IMG_STRIDE
    w = wh[b, 0, r, cc].float().clamp(min=0) * IMG_STRIDE
    h = wh[b, 1, r, cc].float().clamp(min=0) * IMG_STRIDE
    boxes = torch.stack([u - w / 2, v - h, u + w / 2, v], 1)
    out.append((boxes.cpu().numpy(), c.cpu().numpy(), s.cpu().numpy()))
  return out


def gt_boxes(heat: torch.Tensor, wh: torch.Tensor, off: torch.Tensor, pos: torch.Tensor):
  out = []
  for b in range(heat.shape[0]):
    r, cc = torch.nonzero(pos[b] > 0, as_tuple=True)
    c = heat[b, :, r, cc].argmax(0)
    u = (cc.float() + off[b, 0, r, cc]) * IMG_STRIDE
    v = (r.float() + off[b, 1, r, cc]) * IMG_STRIDE
    w, h = wh[b, 0, r, cc] * IMG_STRIDE, wh[b, 1, r, cc] * IMG_STRIDE
    out.append((torch.stack([u - w / 2, v - h, u + w / 2, v], 1).cpu().numpy(), c.cpu().numpy()))
  return out


class CocoAccumulator:
  def __init__(self):
    self.images, self.gts, self.dets = [], [], []

  def add(self, pred, gt) -> None:
    img_id = len(self.images) + 1
    self.images.append({"id": img_id, "width": 512, "height": 256})
    for box, c in zip(*gt, strict=True):
      x1, y1, x2, y2 = map(float, box)
      self.gts.append({"id": len(self.gts) + 1, "image_id": img_id, "category_id": int(c) + 1,
                       "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    for box, c, s in zip(*pred, strict=True):
      x1, y1, x2, y2 = map(float, box)
      self.dets.append({"image_id": img_id, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(s)})

  def summary(self) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not self.gts or not self.dets:
      return {"mAP50": 0.0, "mAP": 0.0}
    gt = COCO()
    gt.dataset = {"images": self.images, "annotations": self.gts,
                  "categories": [{"id": i + 1, "name": str(i)} for i in range(N_CLS)]}
    with contextlib.redirect_stdout(io.StringIO()):
      gt.createIndex()
      dt = gt.loadRes(self.dets)
      ev = COCOeval(gt, dt, "bbox")
      ev.params.areaRng = [[0, 1e10], [0, 8**2], [8**2, 24**2], [24**2, 1e10]]  # small cones are tiny at 512x256
      ev.evaluate()
      ev.accumulate()
      ev.summarize()
    s = ev.stats
    per_cls = []
    for ci in range(N_CLS):
      pr = ev.eval["precision"][0, :, ci, 0, -1]
      per_cls.append(float(np.mean(pr[pr > -1])) if (pr > -1).any() else None)
    return {"mAP": float(s[0]), "mAP50": float(s[1]), "mAP_small": float(s[3]), "AP50_per_class": per_cls}
