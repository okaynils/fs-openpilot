"""ONNX -> PyTorch conversion of the (prepared, FP32) vision and policy graphs.

The torch modules are always rebuilt from the ONNX files (a few seconds) and checkpoints store only
``state_dict``s, which keeps them independent of onnx2torch's pickling details.

onnx2torch registers weights consumed by MatMul/Mul/Add (e.g. the policy transformer, layer-scale
gammas) as buffers; ``promote_buffers`` turns float buffers into trainable parameters. Every parameter
gets mapped back to its original ONNX initializer name (``vision_model.vision._en.stages.3...``) so
freezing / LR groups can be defined by readable prefixes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnx2torch import convert as o2t_convert
from torch import nn

import fsop.model.op_converters  # noqa: F401  (registers converters)

NON_TRAINABLE = ("_mean", "_std")


def _hash(arr: np.ndarray) -> str:
  return hashlib.sha1(np.ascontiguousarray(arr, dtype=np.float32).tobytes() + str(arr.shape).encode()).hexdigest()


def promote_buffers(gm: nn.Module) -> None:
  init_mod = getattr(gm, "initializers", None)
  if init_mod is None:
    return
  for name, buf in list(init_mod.named_buffers(recurse=False)):
    if buf.is_floating_point() and buf.numel() > 1:
      delattr(init_mod, name)
      init_mod.register_parameter(name, nn.Parameter(buf.detach().clone()))


def onnx_name_map(gm: nn.Module, model: onnx.ModelProto) -> dict[str, str]:
  """torch parameter name -> ONNX initializer name."""
  inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
  by_hash: dict[str, list[str]] = {}
  for k, v in inits.items():
    if v.dtype.kind == "f":
      by_hash.setdefault(_hash(v), []).append(k)
  node_weights = {}
  for n in model.graph.node:
    ws = [i for i in n.input if i in inits]
    if ws:
      node_weights[n.name] = ws
  out = {}
  for pname, p in gm.named_parameters():
    mod, _, leaf = pname.rpartition(".")
    if mod in node_weights and leaf in ("weight", "bias"):
      ws = node_weights[mod]
      idx = 0 if leaf == "weight" else 1
      if idx < len(ws):
        out[pname] = ws[idx]
        continue
    h = _hash(p.detach().cpu().numpy())
    cands = by_hash.get(h)
    if cands:
      out[pname] = cands[0]
    else:
      out[pname] = pname
  return out


def freeze_non_trainable(gm: nn.Module, names: dict[str, str]) -> None:
  for pname, p in gm.named_parameters():
    if any(names.get(pname, "").endswith(s) for s in NON_TRAINABLE):
      p.requires_grad_(False)


def load_graph(path: str | Path) -> tuple[nn.Module, dict[str, str]]:
  model = onnx.load(str(path))
  gm = o2t_convert(model)
  promote_buffers(gm)
  names = onnx_name_map(gm, model)
  freeze_non_trainable(gm, names)
  return gm, names
