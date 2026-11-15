"""Current-weight, detached visual history for image training.

The prepared policy gathers [-9..-1] from the 25-slot buffer: eight past features
and the current feature. Older slots retain their public shape but are unused.
"""
from __future__ import annotations

import torch

from fsop.model.temporal import N_FEAT

ACTIVE_HISTORY = 8


def validate_policy_history(path) -> None:
  """Fail closed if a replacement policy graph reads more than the supported eight past slots."""
  import numpy as np
  import onnx
  from onnx import numpy_helper

  graph = onnx.load(path).graph
  users = [node for node in graph.node if "fb_full" in node.input]
  if len(users) != 1 or users[0].op_type != "Transpose":
    raise ValueError("Unsupported fresh-history policy input graph")
  transpose = users[0]
  attributes = {a.name: onnx.helper.get_attribute_value(a) for a in transpose.attribute}
  users = [node for node in graph.node if transpose.output[0] in node.input]
  if attributes.get("perm") != [1, 0, 2] or len(users) != 1 or users[0].op_type != "GatherND":
    raise ValueError("Unsupported fresh-history policy gather")
  gather = users[0]
  arrays = {value.name: value for value in graph.initializer}
  attributes = {a.name: onnx.helper.get_attribute_value(a) for a in gather.attribute}
  indices = arrays.get(gather.input[1])
  if (attributes.get("batch_dims", 0) != 0 or indices is None
      or not np.array_equal(numpy_helper.to_array(indices), np.arange(-ACTIVE_HISTORY - 1, 0)[:, None])):
    raise ValueError("Fresh-history policy must gather current plus eight previous features")


@torch.no_grad()
def encode_fresh_history(encode, road: torch.Tensor, wide: torch.Tensor, valid: torch.Tensor,
                         chunk_size: int = 32) -> torch.Tensor:
  """Encode nine recorded frames into eight past pairs, keeping absent past exactly zero."""
  if road.shape != wide.shape or road.ndim != 5 or road.shape[1:3] != (ACTIVE_HISTORY + 1, 6):
    raise ValueError("Fresh history requires matching [B,9,6,H,W] frame windows")
  if valid.shape != (road.shape[0], ACTIVE_HISTORY) or valid.dtype != torch.bool or chunk_size < 1:
    raise ValueError("Invalid fresh-history mask or chunk size")
  past = torch.zeros((road.shape[0], ACTIVE_HISTORY, 512), dtype=torch.float32, device=road.device)
  indices = valid.flatten().nonzero().flatten()
  # Runtime feature queues contain FP32 inference features, not BF16 training activations.
  with torch.autocast(device_type=road.device.type, enabled=False):
    for selected in indices.split(chunk_size):
      if not selected.numel():
        continue
      batch, time = selected // ACTIVE_HISTORY, selected % ACTIVE_HISTORY
      a = torch.cat([road[batch, time], road[batch, time + 1]], dim=1).float()
      b = torch.cat([wide[batch, time], wide[batch, time + 1]], dim=1).float()
      past.reshape(-1, 512)[selected] = encode(a, b).hidden.float()
  result = past.new_zeros((road.shape[0], N_FEAT, 512))
  result[:, -ACTIVE_HISTORY:] = past
  return result
