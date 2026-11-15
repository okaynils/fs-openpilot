"""Export the fine-tuned model to ONNX in openpilot's stateless layout.

Legacy inputs match the source graph (img, big_img uint8 [1,12,128,256], features_buffer,
desire_pulse, traffic_convention, action_t). Conditioned checkpoints additionally require
versioned calibrated geometry; measured-speed checkpoints add speed [1,1] in m/s.
``outputs`` keeps the original 2576 values (vision heads,
hidden_state, plan, desire_state, pad) and appends the BEV cone detections decoded in-graph
(top-K peaks: x, y [ego m, x fwd / y left], class, score). ``output_slices`` metadata is updated, so
openpilot's Parser slices the original fields unchanged.
"""

from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from fsop.data.targets import GRID
from fsop.model.build import SupercomboFS, load_fs

TOPK = 64


class ExportWrapper(nn.Module):
  def __init__(self, m: SupercomboFS, topk: int = TOPK):
    super().__init__()
    self.m = m
    self.topk = topk

  def forward(self, img, big_img, features_buffer, desire_pulse, traffic_convention, action_t, geometry=None,
              speed=None):
    base = self.m.base
    v = base.encode(img.float(), big_img.float())
    pred = self.m.plan(features_buffer, v.hidden, desire_pulse, traffic_convention, geometry, speed)
    plan = torch.cat([pred["plan_mu"].flatten(1), pred["plan_logstd"].flatten(1)], 1)
    desire_state = pred["desire_state"]
    pad = plan.new_zeros(plan.shape[0], 2) + 0 * action_t[:, :1]
    o = self.m.bev(v.s16, v.final)
    p = torch.sigmoid(o["heat"])
    peak = (p == nn.functional.max_pool2d(p, 3, 1, 1)).float() * p
    B, C, H, W = peak.shape
    score, idx = torch.topk(peak.reshape(B, -1), self.topk, dim=1)
    cls = torch.div(idx, H * W, rounding_mode="floor")
    cell = idx - cls * H * W
    r = torch.div(cell, W, rounding_mode="floor")
    c = cell - r * W
    off = torch.sigmoid(o["off"]).reshape(B, 2, -1)
    dr = torch.gather(off[:, 0], 1, cell)
    dc = torch.gather(off[:, 1], 1, cell)
    x = (r.float() + dr) * GRID.cell + GRID.x0
    y = GRID.y - (c.float() + dc) * GRID.cell
    cones = torch.stack([x, y, cls.float(), score], 2).reshape(B, -1)
    outs = [v.raw, plan, desire_state, pad, cones]
    if self.m.img is not None:
      io_ = self.m.img(v.s4, v.s8, v.s16)
      for cam in ("road", "wide"):
        outs.append(self._decode_img(io_[cam]))
    return torch.cat(outs, 1)

  def _decode_img(self, o):
    """CenterNet peaks -> top-K x [u, v (bottom centre, px in the 512x256 slot), w, h, class, score]."""
    from fsop.data.targets import IMG_STRIDE

    p = torch.sigmoid(o["heat"])
    peak = (p == nn.functional.max_pool2d(p, 3, 1, 1)).float() * p
    B, C, H, W = peak.shape
    score, idx = torch.topk(peak.reshape(B, -1), self.topk, dim=1)
    cls = torch.div(idx, H * W, rounding_mode="floor")
    cell = idx - cls * H * W
    r = torch.div(cell, W, rounding_mode="floor")
    c = cell - r * W
    off = torch.sigmoid(o["off"]).reshape(B, 2, -1)
    wh = o["wh"].clamp(min=0).reshape(B, 2, -1)
    u = (c.float() + torch.gather(off[:, 0], 1, cell)) * IMG_STRIDE
    v = (r.float() + torch.gather(off[:, 1], 1, cell)) * IMG_STRIDE
    w = torch.gather(wh[:, 0], 1, cell) * IMG_STRIDE
    h = torch.gather(wh[:, 1], 1, cell) * IMG_STRIDE
    return torch.stack([u, v, w, h, cls.float(), score], 2).reshape(B, -1)


SRC_ONNX = "artifacts/models/driving_supercombo_b361e952.onnx"


def export_onnx(ckpt: str | Path, out: str | Path, src_onnx: str | Path = SRC_ONNX, opset: int = 17) -> Path:
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  m = load_fs(ckpt).eval()
  w = ExportWrapper(m).eval()
  args = (torch.zeros(1, 12, 128, 256, dtype=torch.uint8), torch.zeros(1, 12, 128, 256, dtype=torch.uint8),
          torch.zeros(1, 24, 512), torch.zeros(1, 25, 8), torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2))
  names = ["img", "big_img", "features_buffer", "desire_pulse", "traffic_convention", "action_t"]
  if m.context_adapter is not None:
    from fsop.geometry.camera import CameraMount
    from fsop.model.conditioning import geometry_context
    from fsop.sim.vehicle import VehicleParams
    args += (torch.from_numpy(geometry_context(CameraMount(), VehicleParams(), m.context_version)[None]),)
    names.append("geometry")
  if m.state_mode == "speed":
    args += (torch.ones(1, 1),)
    names.append("speed")
  torch.onnx.export(w, args, str(out), input_names=names, output_names=["outputs"], opset_version=opset,
                    dynamo=False, do_constant_folding=True)
  model = onnx.load(str(out))
  n_out = 2576 + 4 * TOPK + (2 * 6 * TOPK if m.img is not None else 0)
  dims = model.graph.output[0].type.tensor_type.shape.dim
  dims[0].dim_value, dims[1].dim_value = 1, n_out
  src = onnx.load(str(src_onnx), load_external_data=False)
  meta = {p.key: p.value for p in src.metadata_props}
  slices = pickle.loads(base64.b64decode(meta["output_slices"]))
  slices["pad"] = slice(2574, 2576)
  slices["cones_bev"] = slice(2576, 2576 + 4 * TOPK)
  if m.img is not None:
    a = 2576 + 4 * TOPK
    slices["cones_img_road"] = slice(a, a + 6 * TOPK)
    slices["cones_img_wide"] = slice(a + 6 * TOPK, a + 12 * TOPK)
    meta["fsop_cones_img_layout"] = f"top{TOPK} x [u, v (bottom centre px, 512x256 slot), w, h, class, score]"
  meta["output_slices"] = base64.b64encode(pickle.dumps(slices)).decode()
  if m.context_adapter is not None:
    from fsop.model.conditioning import context_fields, context_scale
    meta["fsop_context_fields"] = json.dumps(list(context_fields(m.context_version)))
    meta["fsop_context_scale"] = json.dumps(context_scale(m.context_version).tolist())
  meta["fsop_context_version"] = str(m.context_version) if m.context_adapter is not None else "0"
  meta["fsop_context_scope"] = m.context_scope
  meta["fsop_state_mode"] = m.state_mode
  meta["fsop_state_version"] = "1"
  meta["fsop_speed_scale_mps"] = "15.0"
  meta["fsop_trajectory_origin"] = "ground_projected_cog"
  if m.context_anchor is not None:
    meta["fsop_context_anchor"] = json.dumps(m.context_anchor.cpu().tolist())
  meta["fsop_checkpoint"] = str(ckpt)
  meta["fsop_cones_bev_layout"] = f"top{TOPK} x [x_m, y_m (left+), class(0 blue,1 yellow,2 orange,3 big_orange), score]"
  del model.metadata_props[:]
  for k, v in meta.items():
    model.metadata_props.add(key=k, value=v)
  onnx.save(model, str(out))
  _check(w, out, args)
  return out


def _check(w: nn.Module, path: Path, args) -> None:
  import onnxruntime as ort

  rng = np.random.default_rng(0)
  a = list(args)
  a[0] = torch.from_numpy(rng.integers(30, 220, (1, 12, 128, 256), dtype=np.uint8))
  a[1] = torch.from_numpy(rng.integers(30, 220, (1, 12, 128, 256), dtype=np.uint8))
  with torch.no_grad():
    ref = w(*a).numpy()
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  feed = {i.name: x.numpy() for i, x in zip(sess.get_inputs(), a, strict=True)}
  got = sess.run(None, feed)[0]
  err = np.abs(got[:, :2576] - ref[:, :2576]).max()
  print(f"export parity (ORT vs torch, original outputs): max abs {err:.2e}")
  if err > 1e-2:
    raise RuntimeError("ONNX export parity check failed")
