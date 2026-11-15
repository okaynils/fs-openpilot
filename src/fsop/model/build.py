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
               anchored_context: bool = False, context_version: int = 1, context_scope: str = "current",
               state_mode: str = "none"):
    super().__init__()
    if state_mode not in ("none", "speed", "constant") or (state_mode != "none" and not conditioned):
      raise ValueError("Unsupported state mode or missing calibrated geometry")
    self.state_mode = state_mode
    if context_scope not in ("current", "sequence"):
      raise ValueError(f"Unsupported conditioning scope: {context_scope}")
    self.context_scope = context_scope
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
    self.motion_adapter = None
    if state_mode != "none":
      from fsop.model.conditioning import context_fields
      self.motion_adapter = nn.Sequential(nn.Linear(513 + len(context_fields(context_version)), 128), nn.SiLU(),
                                          nn.Linear(128, 512))
      nn.init.zeros_(self.motion_adapter[-1].weight)
      nn.init.zeros_(self.motion_adapter[-1].bias)
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
           traffic: torch.Tensor, geometry=None, speed=None):
    fb_full = torch.cat([history, hidden[:, None]], dim=1)
    if self.context_adapter is not None:
      if geometry is None:
        raise ValueError(f"This checkpoint requires calibrated geometry context v{self.context_version}")
      features = fb_full if self.context_scope == "sequence" else hidden
      context = geometry[:, None].expand(-1, features.shape[1], -1) if features.ndim == 3 else geometry
      correction = self.context_adapter(torch.cat([features, context], dim=-1))
      if self.context_anchor is not None:
        anchor = self.context_anchor.expand_as(context)
        correction = correction - self.context_adapter(torch.cat([features, anchor], dim=-1))
      if self.context_scope == "sequence":
        # Keep zero padding as absent history. Current features always represent an observation.
        present = torch.cat([history.abs().sum(-1, keepdim=True) > 0,
                             torch.ones_like(hidden[:, None, :1], dtype=torch.bool)], dim=1)
        fb_full = fb_full + correction * present.to(correction.dtype)
      else:
        fb_full = torch.cat([history, (hidden + correction)[:, None]], dim=1)
    if self.motion_adapter is not None:
      if self.state_mode == "speed" and speed is None:
        raise ValueError("This checkpoint requires measured forward speed in m/s")
      state = (speed.reshape(-1, 1).to(hidden) / 15. if self.state_mode == "speed"
               else hidden.new_zeros((hidden.shape[0], 1)))
      correction = self.motion_adapter(torch.cat([hidden, geometry, state], -1))
      fb_full = torch.cat([fb_full[:, :-1], (fb_full[:, -1] + correction)[:, None]], 1)
    plan, desire_state = self.base.policy_raw(fb_full, desire, traffic)
    mu, log_std = split_plan(plan)
    return {"plan_mu": mu, "plan_logstd": log_std, "desire_state": desire_state}

  def forward_cached(self, b: dict) -> dict:
    out = self.plan(b["history"], b["hidden"], b["desire"], b["traffic"], b.get("geometry"), b.get("speed"))
    out["bev"] = self.bev(b["s16"], b["final"])
    return out

  def forward_images(self, b: dict, heads: tuple[str, ...] = ("plan", "bev", "img")) -> dict:
    v = self.base.encode(b["img"].float(), b["big_img"].float())
    out: dict = {"hidden": v.hidden}
    if "plan" in heads:
      history = b["history"]
      if "history_frames_road" in b:
        from fsop.model.history import encode_fresh_history
        history = encode_fresh_history(self.base.encode, b["history_frames_road"],
                                       b["history_frames_wide"], b["history_valid"])
      out.update(self.plan(history, v.hidden, b["desire"], b["traffic"], b.get("geometry"), b.get("speed")))
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
  contract = sd.get("input_contract", {})
  state_mode = contract.get("state_mode", "none")
  has_motion = any(k.startswith("motion_adapter.") for k in sd["model"])
  if state_mode not in ("none", "speed", "constant") or has_motion != (state_mode != "none"):
    raise ValueError("Checkpoint motion weights disagree with state contract")
  if has_motion and (contract.get("state_version") != 1 or contract.get("speed_scale_mps") != 15.):
    raise ValueError("Unsupported motion state contract")
  conditioned = any(k.startswith("context_adapter.") for k in sd["model"])
  version = 1
  scope = "current"
  if conditioned:
    from fsop.model.conditioning import context_fields
    contract = sd.get("input_contract", {})
    version = contract.get("geometry_version")
    scope = contract.get("conditioning_scope", "current")
    if version not in (1, 2) or contract.get("fields") != list(context_fields(version)):
      raise ValueError("Unsupported or missing conditioned checkpoint input contract")
    if ("context_anchor" in sd["model"]) != bool(contract.get("anchored_context", False)):
      raise ValueError("Checkpoint adapter disagrees with its anchoring contract")
  m = SupercomboFS(image_head=any(k.startswith("img.") for k in sd["model"]),
                   conditioned=conditioned, anchored_context="context_anchor" in sd["model"], context_version=version,
                   context_scope=scope, state_mode=state_mode)
  m.load_state_dict(sd["model"], strict=True)
  return m
