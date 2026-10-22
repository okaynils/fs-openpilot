"""Export the fine-tuned model to ONNX in openpilot's stateless layout.

Inputs are identical to the source graph (img, big_img uint8 [1,12,128,256], features_buffer,
desire_pulse, traffic_convention, action_t). ``outputs`` keeps the original 2576 values (vision heads,
hidden_state, plan, desire_state, pad) and appends the BEV cone detections decoded in-graph
(top-K peaks: x, y [ego m, x fwd / y left], class, score). ``output_slices`` metadata is updated, so
openpilot's Parser slices the original fields unchanged.
"""

from __future__ import annotations

import base64
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

  def forward(self, img, big_img, features_buffer, desire_pulse, traffic_convention, action_t):
    base = self.m.base
    v = base.encode(img.float(), big_img.float())
    fb_full = torch.cat([features_buffer, v.hidden[:, None]], 1)
    plan, desire_state = base.policy_raw(fb_full, desire_pulse, traffic_convention)
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
    return torch.cat([v.raw, plan, desire_state, pad, cones], 1)


SRC_ONNX = "artifacts/models/driving_supercombo_b361e952.onnx"


def export_onnx(ckpt: str | Path, out: str | Path, src_onnx: str | Path = SRC_ONNX, opset: int = 17) -> Path:
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  m = load_fs(ckpt).eval()
  w = ExportWrapper(m).eval()
  args = (torch.zeros(1, 12, 128, 256, dtype=torch.uint8), torch.zeros(1, 12, 128, 256, dtype=torch.uint8),
          torch.zeros(1, 24, 512), torch.zeros(1, 25, 8), torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2))
  names = ["img", "big_img", "features_buffer", "desire_pulse", "traffic_convention", "action_t"]
  torch.onnx.export(w, args, str(out), input_names=names, output_names=["outputs"], opset_version=opset,
                    dynamo=False, do_constant_folding=True)
  model = onnx.load(str(out))
  dims = model.graph.output[0].type.tensor_type.shape.dim
  dims[0].dim_value, dims[1].dim_value = 1, 2576 + 4 * TOPK
  src = onnx.load(str(src_onnx), load_external_data=False)
  meta = {p.key: p.value for p in src.metadata_props}
  slices = pickle.loads(base64.b64decode(meta["output_slices"]))
  slices["pad"] = slice(2574, 2576)
  slices["cones_bev"] = slice(2576, 2576 + 4 * TOPK)
  meta["output_slices"] = base64.b64encode(pickle.dumps(slices)).decode()
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
