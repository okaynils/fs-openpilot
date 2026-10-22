"""Losses: Laplacian NLL for the plan (openpilot-style), CenterNet focal + L1 for cone heads."""

from __future__ import annotations

import torch

from fsop.sim.expert import T_IDXS

# per-column weights: position, velocity, acceleration, euler, orientation rate (x3 each)
PLAN_COL_W = torch.tensor([1.0, 1.0, 0.1, 0.5, 0.5, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3])
# per-time weights: full weight up to 4 s, fading to 0.2 at 10 s (far beyond the cones' visibility)
PLAN_T_W = torch.tensor([1.0 if t <= 4.0 else max(0.2, 1.0 - 0.8 * (t - 4.0) / 6.0) for t in T_IDXS])


def plan_nll(mu: torch.Tensor, log_std: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Laplace NLL |y-mu|/b + log(2b), b = exp(log_std); mu/y [B,33,15]."""
  log_b = log_std.clamp(-7.0, 7.0)
  nll = (y - mu).abs() * torch.exp(-log_b) + log_b
  w = PLAN_T_W.to(mu)[None, :, None] * PLAN_COL_W.to(mu)[None, None, :]
  return (nll * w).sum((1, 2)).mean() / w.sum()


def plan_l1(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  w = PLAN_T_W.to(mu)[None, :, None] * PLAN_COL_W.to(mu)[None, None, :]
  return ((mu - y).abs() * w).sum((1, 2)).mean() / w.sum()


def focal_heatmap(logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None, alpha: float = 2.0,
                  beta: float = 4.0) -> torch.Tensor:
  """CenterNet penalty-reduced focal loss. logits/gt [B,C,H,W]; mask [B,H,W] (1 = supervise)."""
  p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
  pos = gt.eq(1).float()
  neg = 1.0 - pos
  pos_l = torch.log(p) * (1 - p) ** alpha * pos
  neg_l = torch.log(1 - p) * p**alpha * (1 - gt) ** beta * neg
  if mask is not None:
    m = mask[:, None].float()
    pos_l, neg_l = pos_l * m, neg_l * m
  n_pos = pos.sum().clamp(min=1.0)
  return -(pos_l.sum() + neg_l.sum()) / n_pos


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
  """pred/gt [B,K,H,W], pos [B,H,W]."""
  m = pos[:, None].float()
  return ((pred - gt).abs() * m).sum() / (m.sum() * pred.shape[1]).clamp(min=1.0)


def bev_loss(out: dict, b: dict) -> dict[str, torch.Tensor]:
  heat = focal_heatmap(out["heat"], b["bev_heat"], b["bev_mask"])
  off = masked_l1(torch.sigmoid(out["off"]), b["bev_off"], b["bev_pos"])
  return {"bev_heat": heat, "bev_off": off}


def img_loss(out: dict, b: dict, cams=("road", "wide")) -> dict[str, torch.Tensor]:
  res = {}
  for c in cams:
    if f"{c}_heat" not in b:
      continue
    o = out[c]
    valid = b.get(f"{c}_valid")  # per-sample mask for FSOCO samples without that slot
    m = None if valid is None else valid[:, None, None].float().expand_as(b[f"{c}_pos"])
    res[f"img_{c}_heat"] = focal_heatmap(o["heat"], b[f"{c}_heat"], m)
    res[f"img_{c}_wh"] = 0.1 * masked_l1(o["wh"], b[f"{c}_wh"], b[f"{c}_pos"])
    res[f"img_{c}_off"] = masked_l1(torch.sigmoid(o["off"]), b[f"{c}_off"], b[f"{c}_pos"])
  return res
