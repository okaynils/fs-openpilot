"""Decode BEV cone heatmaps and score them (Hungarian matching per class)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from fsop.data.targets import GRID, N_CLS


def decode_bev(heat_logits: torch.Tensor, off_logits: torch.Tensor, thresh: float = 0.3, k: int = 96):
  """[B,C,H,W] -> list (per batch) of (xy [N,2] ego, cls [N], score [N])."""
  p = torch.sigmoid(heat_logits.float())
  peak = (p == torch.nn.functional.max_pool2d(p, 3, 1, 1)) & (p > thresh)
  off = torch.sigmoid(off_logits.float())
  out = []
  for bi in range(p.shape[0]):
    c, r, cc = torch.nonzero(peak[bi], as_tuple=True)
    s = p[bi, c, r, cc]
    if len(s) > k:
      top = torch.topk(s, k).indices
      c, r, cc, s = c[top], r[top], cc[top], s[top]
    dr, dc = off[bi, 0, r, cc], off[bi, 1, r, cc]
    rc = torch.stack([r.float() + dr, cc.float() + dc], 1).cpu().numpy()
    out.append((GRID.from_cell(rc), c.cpu().numpy(), s.cpu().numpy()))
  return out


def gt_from_targets(heat: torch.Tensor, off: torch.Tensor, pos: torch.Tensor):
  out = []
  for bi in range(heat.shape[0]):
    r, cc = torch.nonzero(pos[bi] > 0, as_tuple=True)
    cls = heat[bi, :, r, cc].argmax(0)
    rc = torch.stack([r.float() + off[bi, 0, r, cc], cc.float() + off[bi, 1, r, cc]], 1).cpu().numpy()
    out.append((GRID.from_cell(rc), cls.cpu().numpy()))
  return out


class BevScorer:
  """Accumulates TP/FP/FN per class and range bin, position error and blue<->yellow swaps."""

  BINS = ((0, 10), (10, 20), (20, 45))

  def __init__(self, gate: float = 0.75):
    self.gate = gate
    self.tp = np.zeros((N_CLS, len(self.BINS)))
    self.fp = np.zeros((N_CLS, len(self.BINS)))
    self.fn = np.zeros((N_CLS, len(self.BINS)))
    self.err: list[float] = []
    self.confusion = np.zeros((N_CLS, N_CLS))

  def _bin(self, xy):
    d = np.linalg.norm(xy, axis=1)
    return np.clip(np.digitize(d, [b[1] for b in self.BINS[:-1]]), 0, len(self.BINS) - 1)

  def add(self, pred_xy, pred_cls, gt_xy, gt_cls):
    # class-agnostic matching for the confusion matrix
    if len(pred_xy) and len(gt_xy):
      D = np.linalg.norm(pred_xy[:, None] - gt_xy[None], axis=2)
      pi, gi = linear_sum_assignment(D)
      for a, b in zip(pi, gi, strict=True):
        if D[a, b] < self.gate:
          self.confusion[gt_cls[b], pred_cls[a]] += 1
    for c in range(N_CLS):
      P, G = pred_xy[pred_cls == c], gt_xy[gt_cls == c]
      matched_p, matched_g = np.zeros(len(P), bool), np.zeros(len(G), bool)
      if len(P) and len(G):
        D = np.linalg.norm(P[:, None] - G[None], axis=2)
        pi, gi = linear_sum_assignment(D)
        ok = D[pi, gi] < self.gate
        matched_p[pi[ok]] = True
        matched_g[gi[ok]] = True
        self.err.extend(D[pi[ok], gi[ok]].tolist())
      if len(P):
        np.add.at(self.tp[c], self._bin(P[matched_p]), 1) if matched_p.any() else None
        np.add.at(self.fp[c], self._bin(P[~matched_p]), 1) if (~matched_p).any() else None
      if len(G) and (~matched_g).any():
        np.add.at(self.fn[c], self._bin(G[~matched_g]), 1)

  def summary(self) -> dict:
    tp, fp, fn = self.tp.sum(1), self.fp.sum(1), self.fn.sum(1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    by_range = {}
    for j, (a, b) in enumerate(self.BINS):
      t, p_, n = self.tp[:, j].sum(), self.fp[:, j].sum(), self.fn[:, j].sum()
      pr, rc = t / max(t + p_, 1), t / max(t + n, 1)
      by_range[f"{a}-{b}m"] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9)}
    T, P_, N = tp.sum(), fp.sum(), fn.sum()
    pr, rc = T / max(T + P_, 1), T / max(T + N, 1)
    by = self.confusion[:2, :2]
    return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9),
            "per_class_f1": f1.tolist(), "pos_err_m": float(np.mean(self.err)) if self.err else None,
            "blue_yellow_swap_rate": float((by[0, 1] + by[1, 0]) / max(by.sum(), 1)), "by_range": by_range}
