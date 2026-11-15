"""Deployment runtime: raw camera images in, plan + cones + control commands out, at 20 Hz.

    from fsop.runtime import FSDriver
    drv = FSDriver("artifacts/export/fs_supercombo.onnx", profile="configs/profiles/my_team.yaml")
    while running:                                   # call at 20 Hz (the model's rate)
        out = drv.step({"front": rgb}, v_ego=wheel_speed)
        send(out.steer, out.accel)                   # or feed out.plan / out.cones to your own stack

Backends: an exported ONNX file (``fsop export``; only numpy + onnxruntime + opencv needed on the car)
or a training checkpoint (``.pt``, needs torch). The runtime keeps openpilot's temporal queues
(frames t-4/t, 5 Hz feature history, desire pulses), warps the team's cameras into the model frame
(``fsop.geometry.warp``) and projects image-head cone detections onto the ground with the profile's
mount. The mission layer (launch / lap counting / stopping) is *not* included - that belongs to the
car's AS state machine.
"""

from __future__ import annotations

import base64
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fsop.geometry.op_frames import op_to_ego_plan
from fsop.geometry.warp import ModelFrameWarper
from fsop.geometry.yuv import rgb_to_model
from fsop.model.temporal import TemporalState


@dataclass
class DriverOutput:
  plan: np.ndarray  # [33,15] ego frame (x fwd, y left, z up), times fsop.sim.expert.T_IDXS
  plan_std: np.ndarray  # [33,15]
  cones: np.ndarray  # [N,4] x, y (ego, m), class (0 blue, 1 yellow, 2 orange, 3 big orange), score
  steer: float  # front wheel angle [rad], left positive
  accel: float  # [m/s^2]
  extra: dict = field(default_factory=dict)


def _speed_input(speed) -> np.ndarray:
  if speed is None or not np.isscalar(speed) or not np.isfinite(speed):
    raise ValueError("Measured forward speed must be a finite scalar in m/s")
  return np.array([[speed]], np.float32)


class OnnxEngine:
  def __init__(self, path: str | Path, providers: list[str] | None = None):
    import onnx
    import onnxruntime as ort

    meta = {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}
    version = meta.get("fsop_context_version", "0")
    if version not in ("0", "1", "2"):
      raise ValueError(f"Unsupported geometry contract version: {version}")
    self.context_scope = meta.get("fsop_context_scope", "current")
    if self.context_scope not in ("current", "sequence"):
      raise ValueError("Unsupported conditioning scope")
    self.context_version = int(version)
    self.state_mode = meta.get("fsop_state_mode", "none")
    if self.state_mode not in ("none", "speed", "constant"):
      raise ValueError("Unsupported motion state mode")
    if self.state_mode != "none" and (meta.get("fsop_state_version") != "1"
                                      or meta.get("fsop_speed_scale_mps") != "15.0"):
      raise ValueError("Unsupported motion state contract")
    self.slices = pickle.loads(base64.b64decode(meta["output_slices"]))
    if providers is None:  # CoreML/TensorRT fp16 paths drift on the long plan horizon -> opt-in only
      avail = ort.get_available_providers()
      providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    providers = [("CUDAExecutionProvider", {"use_tf32": 0}) if p == "CUDAExecutionProvider" else p
                 for p in providers]
    self.sess = ort.InferenceSession(str(path), providers=providers)
    self.types = {i.name: i.type for i in self.sess.get_inputs()}
    if (self.state_mode == "speed") != ("speed" in self.types):
      raise ValueError("ONNX speed input disagrees with state contract")
    if self.state_mode == "speed" and next(i for i in self.sess.get_inputs() if i.name == "speed").shape != [1, 1]:
      raise ValueError("Unsupported ONNX speed shape")
    if (version != "0") != ("geometry" in self.types):
      raise ValueError("ONNX geometry input disagrees with its versioned contract")
    if version != "0":
      from fsop.model.conditioning import context_fields
      fields = context_fields(self.context_version)
      geometry_input = next(i for i in self.sess.get_inputs() if i.name == "geometry")
      if geometry_input.shape != [1, len(fields)]:
        raise ValueError("Unsupported ONNX geometry shape")
      if version == "2" and json.loads(meta.get("fsop_context_fields", "[]")) != list(fields):
        raise ValueError("ONNX geometry fields disagree with context v2")

  def __call__(self, img, big, feats, desire, traffic, geometry=None, speed=None) -> dict[str, np.ndarray]:
    def f(x, name):
      return x.astype(np.float16) if "float16" in self.types[name] else x.astype(np.float32)

    feed = {"img": img[None], "big_img": big[None], "features_buffer": f(feats[None], "features_buffer"),
            "desire_pulse": f(desire[None], "desire_pulse"),
            "traffic_convention": f(traffic[None], "traffic_convention"), "action_t": f(np.zeros((1, 2)), "action_t")}
    if "geometry" in self.types:
      if geometry is None:
        raise ValueError(f"Export requires calibrated geometry context v{self.context_version}")
      feed["geometry"] = f(geometry[None], "geometry")
    if self.state_mode == "speed":
      feed["speed"] = f(_speed_input(speed), "speed")
    out = self.sess.run(None, feed)[0][0].astype(np.float32)
    return {k: out[s] for k, s in self.slices.items()}


class TorchEngine:
  def __init__(self, ckpt: str | Path, device: str = "auto"):
    import torch

    from fsop.model.build import load_fs
    from fsop.model.export import TOPK, ExportWrapper
    from fsop.train.device import pick_device, strict_inference_precision

    self.torch = torch
    self.device = pick_device(device)
    strict_inference_precision()
    self.m = ExportWrapper(load_fs(ckpt)).to(self.device).eval()
    self.state_mode = self.m.m.state_mode
    self.context_version = self.m.m.context_version if self.m.m.context_adapter is not None else 0
    n = 2576 + 4 * TOPK
    self.slices = {"hidden_state": slice(1064, 1576), "plan": slice(1576, 2566), "cones_bev": slice(2576, n)}
    if self.m.m.img is not None:
      self.slices.update(cones_img_road=slice(n, n + 6 * TOPK), cones_img_wide=slice(n + 6 * TOPK, n + 12 * TOPK))

  def __call__(self, img, big, feats, desire, traffic, geometry=None, speed=None) -> dict[str, np.ndarray]:
    t = self.torch
    with t.no_grad():
      args = [t.from_numpy(x[None]).to(self.device) for x in (img, big)]
      args += [t.from_numpy(np.asarray(x[None], np.float32)).to(self.device) for x in (feats, desire, traffic)]
      args.append(t.zeros(1, 2, device=self.device))
      if self.m.m.context_adapter is not None:
        if geometry is None:
          raise ValueError(f"Checkpoint requires calibrated geometry context v{self.context_version}")
        args.append(t.from_numpy(geometry[None]).to(self.device))
      if self.state_mode == "speed":
        args.append(t.from_numpy(_speed_input(speed)).to(self.device))
      out = self.m(*args)[0].cpu().numpy()
    return {k: out[s] for k, s in self.slices.items()}


class FSDriver:
  def __init__(self, model: str | Path, profile=None, controller: str = "pp", min_score: float = 0.3,
               providers: list[str] | None = None):
    from fsop.eval.controller import make_controller
    from fsop.profile import TeamProfile, load_profile

    self.profile = profile if isinstance(profile, TeamProfile) else load_profile(profile)
    self.engine = OnnxEngine(model, providers) if str(model).endswith(".onnx") else TorchEngine(model)
    self.warper = ModelFrameWarper(self.profile)
    self.ts = TemporalState()
    self.controller_name = controller
    self.ctrl = make_controller(controller, self.profile.vehicle)
    self.min_score = min_score
    from fsop.model.conditioning import geometry_context
    self.geometry = geometry_context(self.profile.mount, self.profile.vehicle, self.engine.context_version or 1)
    self.T_ego_cam = self.profile.mount.T_ego_cam()

  def reset(self) -> None:
    self.ts = TemporalState()
    from fsop.eval.controller import make_controller
    self.ctrl = make_controller(self.controller_name, self.profile.vehicle)

  def step(self, images: dict[str, np.ndarray], v_ego: float, desire: np.ndarray | None = None,
           already_model_frame: bool = False) -> DriverOutput:
    """images: physical camera name -> RGB uint8 (or slot name -> 256x512 RGB if ``already_model_frame``)."""
    state = {"speed": float(_speed_input(v_ego)[0, 0])} if self.engine.state_mode == "speed" else {}
    slots = images if already_model_frame else self.warper(images)
    pair = self.ts.push_frames({s: rgb_to_model(im) for s, im in slots.items()})
    des = self.ts.push_desire(np.zeros(8, np.float32) if desire is None else desire.astype(np.float32))
    out = self.engine(pair["road"], pair["wide"], self.ts.features(), des, np.array([1.0, 0.0], np.float32),
                      self.geometry, **state)
    self.ts.push_hidden(out["hidden_state"])
    p = out["plan"].reshape(2, 33, 15)
    plan = op_to_ego_plan(p[0])
    std = np.exp(np.clip(p[1], -7, 7))
    cones = self._cones(out)
    steer, accel = self.ctrl(plan, v_ego)
    return DriverOutput(plan, std, cones, steer, accel, {"raw": out})

  def _cones(self, out: dict) -> np.ndarray:
    from fsop.eval.img_to_bev import ground_points

    if "cones_img_road" not in out:  # older export: learned BEV head only
      c = out["cones_bev"].reshape(-1, 4)
      return c[c[:, 3] > self.min_score]
    res = []
    for slot, max_r in (("wide", 22.0), ("road", 45.0)):
      d = out[f"cones_img_{slot}"].reshape(-1, 6)
      d = d[d[:, 5] > self.min_score]
      xy, ok = ground_points(d[:, :2], slot, self.T_ego_cam)
      ok &= np.linalg.norm(xy, axis=1) < max_r
      res.append(np.c_[xy[ok], d[ok, 4], d[ok, 5]])
    c = np.concatenate(res) if res else np.zeros((0, 4))
    keep = []  # merge duplicates seen by both slots
    for i in np.argsort(-c[:, 3]):
      if all(c[j, 2] != c[i, 2] or np.hypot(*(c[j, :2] - c[i, :2])) > 0.6 for j in keep):
        keep.append(i)
    return c[keep]
