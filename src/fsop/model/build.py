"""Model construction: original supercombo (converted) + Formula Student heads."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from fsop.model.convert import load_graph
from fsop.model.heads.cone_bev import ConeBEVHead
from fsop.model.heads.cone_img import ConeImageHead
from fsop.model.supercombo import SupercomboBase, split_plan

PREP = Path("artifacts/models/prep")

# trunk parameters that stage 2 unfreezes (ONNX initializer name substrings)
TRUNK_TRAINABLE = ("_en.stages.3", "_en.final_conv", "_en.head", "policy.summarizer", "summarizer.")


def _ensure_prepared(prep: Path = PREP) -> Path:
  if not (prep / "vision_fp32.onnx").exists():
    from fsop.model.fetch import fetch
    from fsop.model.onnx_prep import prepare

    prepare(fetch(), prep)
  return prep


class SupercomboFS(nn.Module):
  """openpilot supercombo + cone BEV head + cone image head."""

  def __init__(self, prep: Path = PREP, image_head: bool = True):
    super().__init__()
    prep = _ensure_prepared(prep)
    vision, self.vision_names = load_graph(prep / "vision_fp32.onnx")
    policy, self.policy_names = load_graph(prep / "policy_fp32.onnx")
    self.base = SupercomboBase(vision, policy)
    self.bev = ConeBEVHead()
    self.img = ConeImageHead() if image_head else None

  # -- parameter groups ---------------------------------------------------------------------
  def trunk_params(self, trainable_only: bool = False):
    for n, p in self.base.vision.named_parameters():
      onnx_name = self.vision_names.get(n, n)
      if trainable_only and not any(s in onnx_name for s in TRUNK_TRAINABLE):
        continue
      yield n, p

  def set_trunk_trainable(self, mode: str) -> None:
    """mode: frozen | last | all"""
    for n, p in self.base.vision.named_parameters():
      onnx_name = self.vision_names.get(n, n)
      fixed = onnx_name.endswith(("_mean", "_std"))
      if mode == "frozen" or fixed:
        p.requires_grad_(False)
      elif mode == "last":
        p.requires_grad_(any(s in onnx_name for s in TRUNK_TRAINABLE))
      else:
        p.requires_grad_(True)

  # -- forward ------------------------------------------------------------------------------
  def plan(self, history: torch.Tensor, hidden: torch.Tensor, desire: torch.Tensor, traffic: torch.Tensor):
    fb_full = torch.cat([history, hidden[:, None]], dim=1)
    plan, desire_state = self.base.policy_raw(fb_full, desire, traffic)
    mu, log_std = split_plan(plan)
    return {"plan_mu": mu, "plan_logstd": log_std, "desire_state": desire_state}

  def forward_cached(self, b: dict) -> dict:
    out = self.plan(b["history"], b["hidden"], b["desire"], b["traffic"])
    out["bev"] = self.bev(b["s16"], b["final"])
    return out

  def forward_images(self, b: dict, heads: tuple[str, ...] = ("plan", "bev", "img")) -> dict:
    v = self.base.encode(b["img"].float(), b["big_img"].float())
    out: dict = {"hidden": v.hidden}
    if "plan" in heads:
      out.update(self.plan(b["history"], v.hidden, b["desire"], b["traffic"]))
    if "bev" in heads:
      out["bev"] = self.bev(v.s16, v.final)
    if "img" in heads and self.img is not None:
      out["img"] = self.img(v.s4, v.s8, v.s16)
    return out


def build_base(ckpt: str | Path | None = None) -> SupercomboBase:
  """The plain (or fine-tuned) supercombo graph pair for caching / baselines."""
  if ckpt is None:
    prep = _ensure_prepared()
    v, _ = load_graph(prep / "vision_fp32.onnx")
    p, _ = load_graph(prep / "policy_fp32.onnx")
    return SupercomboBase(v, p)
  return load_fs(ckpt).base


def load_fs(ckpt: str | Path, map_location: str = "cpu") -> SupercomboFS:
  sd = torch.load(ckpt, map_location=map_location, weights_only=False)
  m = SupercomboFS(image_head=any(k.startswith("img.") for k in sd["model"]))
  m.load_state_dict(sd["model"], strict=True)
  return m
