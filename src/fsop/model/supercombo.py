"""Torch wrappers around the converted supercombo graphs.

``SupercomboBase`` reproduces the original model exactly (used for parity checks and as the
untouched baseline). ``SupercomboFS`` (in ``fsop.model.fs_model``) adds the Formula Student heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# output layout of the original model (from ONNX metadata ``output_slices``)
HIDDEN = slice(1064, 1576)
PLAN = slice(1576, 2566)
VISION_LEN = 1576
FEATURE_LEN = 512
HISTORY = 25  # policy input slots (24 past + current); the graph only reads the last 9
POLICY_CONTEXT = 9
PLAN_T = 33
PLAN_W = 15


@dataclass
class VisionOut:
  raw: torch.Tensor  # [B,1576]
  hidden: torch.Tensor  # [B,512] (L2-normalized)
  s4: torch.Tensor  # [B,64,32,64]
  s8: torch.Tensor  # [B,128,16,32]
  s16: torch.Tensor  # [B,256,8,16]
  s32: torch.Tensor  # [B,512,4,8]
  final: torch.Tensor  # [B,1024,4,8]


class SupercomboBase(nn.Module):
  def __init__(self, vision: nn.Module, policy: nn.Module):
    super().__init__()
    self.vision = vision
    self.policy = policy

  def encode(self, img: torch.Tensor, big_img: torch.Tensor) -> VisionOut:
    raw, s4, s8, s16, s32, final = self.vision(img, big_img)
    return VisionOut(raw, raw[:, HIDDEN], s4, s8, s16, s32, final)

  def policy_raw(self, fb_full: torch.Tensor, desire: torch.Tensor, traffic: torch.Tensor):
    """fb_full [B,25,512] (last slot = current hidden) -> plan [B,990], desire_state [B,8]."""
    return self.policy(fb_full, desire, traffic)

  def full_outputs(self, img, big_img, features_buffer, desire, traffic) -> torch.Tensor:
    """Exactly the original 2576-wide output vector."""
    v = self.encode(img, big_img)
    fb_full = torch.cat([features_buffer, v.hidden[:, None]], dim=1)
    plan, desire_state = self.policy_raw(fb_full, desire, traffic)
    pad = plan.new_zeros(plan.shape[0], 2)
    return torch.cat([v.raw, plan, desire_state, pad], dim=1)


def split_plan(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """[B,990] -> mean [B,33,15], log-std [B,33,15]."""
  p = p.reshape(p.shape[0], 2, PLAN_T, PLAN_W)
  return p[:, 0], p[:, 1]
