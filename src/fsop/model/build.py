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

  def __init__(self, prep: Path = PREP, image_head: bool = True, conditioned: bool = False,
               anchored_context: bool = False, context_version: int = 1):
    super().__init__()
    prep = _ensure_prepared(prep)
    vision, self.vision_names = load_graph(prep / "vision_fp32.onnx")
    policy, self.policy_names = load_graph(prep / "policy_fp32.onnx")
    self.base = SupercomboBase(vision, policy)
    self.context_adapter = None
    self.context_version = context_version
    if anchored_context and not conditioned:
      raise ValueError("An anchored adapter requires conditioning")
    self.register_buffer("context_anchor", None)
    if conditioned:
      from fsop.model.conditioning import context_fields
      self.context_adapter = nn.Sequential(nn.Linear(512 + len(context_fields(context_version)), 256), nn.SiLU(),
                                           nn.Linear(256, 512))
      nn.init.zeros_(self.context_adapter[-1].weight)
      nn.init.zeros_(self.context_adapter[-1].bias)
      if anchored_context:
        from fsop.geometry.camera import CameraMount
        from fsop.model.conditioning import geometry_context
        from fsop.sim.vehicle import VehicleParams
        self.context_anchor = torch.from_numpy(geometry_context(CameraMount(z=1.1), VehicleParams(), context_version))
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
  def plan(self, history: torch.Tensor, hidden: torch.Tensor, desire: torch.Tensor,
           traffic: torch.Tensor, geometry=None):
    if self.context_adapter is not None:
      if geometry is None:
        raise ValueError(f"This checkpoint requires calibrated geometry context v{self.context_version}")
      correction = self.context_adapter(torch.cat([hidden, geometry], dim=-1))
      if self.context_anchor is not None:
        anchor = self.context_anchor.expand_as(geometry)
        correction = correction - self.context_adapter(torch.cat([hidden, anchor], dim=-1))
      hidden = hidden + correction
    fb_full = torch.cat([history, hidden[:, None]], dim=1)
    plan, desire_state = self.base.policy_raw(fb_full, desire, traffic)
    mu, log_std = split_plan(plan)
    return {"plan_mu": mu, "plan_logstd": log_std, "desire_state": desire_state}

  def forward_cached(self, b: dict) -> dict:
    out = self.plan(b["history"], b["hidden"], b["desire"], b["traffic"], b.get("geometry"))
    out["bev"] = self.bev(b["s16"], b["final"])
    return out

  def forward_images(self, b: dict, heads: tuple[str, ...] = ("plan", "bev", "img")) -> dict:
    v = self.base.encode(b["img"].float(), b["big_img"].float())
    out: dict = {"hidden": v.hidden}
    if "plan" in heads:
      out.update(self.plan(b["history"], v.hidden, b["desire"], b["traffic"], b.get("geometry")))
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
  conditioned = any(k.startswith("context_adapter.") for k in sd["model"])
  version = 1
  if conditioned:
    from fsop.model.conditioning import context_fields
    contract = sd.get("input_contract", {})
    version = contract.get("geometry_version")
    if version not in (1, 2) or contract.get("fields") != list(context_fields(version)):
      raise ValueError("Unsupported or missing conditioned checkpoint input contract")
    if ("context_anchor" in sd["model"]) != bool(contract.get("anchored_context", False)):
      raise ValueError("Checkpoint adapter disagrees with its anchoring contract")
  m = SupercomboFS(image_head=any(k.startswith("img.") for k in sd["model"]),
                   conditioned=conditioned, anchored_context="context_anchor" in sd["model"], context_version=version)
  m.load_state_dict(sd["model"], strict=True)
  return m
