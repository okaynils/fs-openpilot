"""Numerical parity: original FP16 ONNX (onnxruntime) vs FP32 ONNX vs converted torch (CPU / MPS)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from fsop.model.convert import load_graph
from fsop.model.onnx_prep import output_slices
from fsop.model.supercombo import SupercomboBase


def random_inputs(seed: int = 0, batch: int = 1) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  # smooth-ish random images (pure noise is far off the training distribution)
  base = rng.integers(40, 200, size=(batch, 12, 16, 32)).astype(np.float32)
  up = np.kron(base, np.ones((1, 1, 8, 8), dtype=np.float32))
  img = np.clip(up + rng.normal(0, 8, up.shape), 0, 255).astype(np.uint8)
  big = np.clip(np.roll(up, 3, axis=3) + rng.normal(0, 8, up.shape), 0, 255).astype(np.uint8)
  fb = rng.normal(0, 1, size=(batch, 24, 512)).astype(np.float32)
  fb /= np.linalg.norm(fb, axis=-1, keepdims=True)
  return {
    "img": img,
    "big_img": big,
    "features_buffer": fb,
    "desire_pulse": np.zeros((batch, 25, 8), np.float32),
    "traffic_convention": np.tile(np.array([[1.0, 0.0]], np.float32), (batch, 1)),
    "action_t": np.zeros((batch, 2), np.float32),
  }


def run_ort(path: str | Path, inputs: dict[str, np.ndarray], fp16: bool) -> np.ndarray:
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  feed = {}
  for i in sess.get_inputs():
    v = inputs[i.name]
    if fp16 and v.dtype == np.float32:
      v = v.astype(np.float16)
    feed[i.name] = v
  return sess.run(["outputs"], feed)[0].astype(np.float32)


def verify(src_onnx: str | Path, prep_dir: str | Path, devices=("cpu", "mps"), out: str | Path | None = None) -> dict:
  prep_dir = Path(prep_dir)
  import onnx

  slices = output_slices(onnx.load(str(src_onnx), load_external_data=False))
  inputs = random_inputs()
  ref16 = run_ort(src_onnx, inputs, fp16=True)
  ref32 = run_ort(prep_dir / "full_fp32.onnx", inputs, fp16=False)
  report: dict = {"fp32_vs_fp16": _cmp(ref32, ref16, slices)}
  vision, _ = load_graph(prep_dir / "vision_fp32.onnx")
  policy, _ = load_graph(prep_dir / "policy_fp32.onnx")
  model = SupercomboBase(vision, policy)
  for dev in devices:
    if dev == "mps" and not torch.backends.mps.is_available():
      continue
    if dev == "cuda" and not torch.cuda.is_available():
      continue
    m = model.to(dev).eval()
    with torch.no_grad():
      t = {k: torch.from_numpy(v).to(dev) for k, v in inputs.items()}
      outp = m.full_outputs(t["img"], t["big_img"], t["features_buffer"], t["desire_pulse"],
                            t["traffic_convention"]).cpu().numpy()
    report[f"torch_{dev}_vs_fp32"] = _cmp(outp, ref32, slices)
    report[f"torch_{dev}_vs_fp16"] = _cmp(outp, ref16, slices)
  if out:
    Path(out).write_text(json.dumps(report, indent=1))
  return report


def _cmp(a: np.ndarray, b: np.ndarray, slices: dict) -> dict:
  res = {"all_max_abs": float(np.abs(a - b).max())}
  for k in ("plan", "hidden_state", "lane_lines", "desire_state"):
    s = slices[k]
    d = np.abs(a[:, s] - b[:, s])
    res[k] = {"max_abs": float(d.max()), "rel": float(d.max() / (np.abs(b[:, s]).max() + 1e-9))}
  return res
