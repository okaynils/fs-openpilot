"""onnx2torch converters for op versions used by supercombo (opset 20) that onnx2torch lacks.

Importing this module registers them (idempotent).
"""

from __future__ import annotations

import torch
from onnx2torch.node_converters.reduce import OnnxReduceStaticAxes
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription, add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import (
  OnnxMapping,
  OperationConverterResult,
  get_const_value,
  onnx_mapping_from_node,
)
from torch import nn


def _alias(op: str, new_version: int, old_version: int) -> None:
  new = OperationDescription("", op, new_version)
  if new in _CONVERTER_REGISTRY:
    return
  _CONVERTER_REGISTRY[new] = _CONVERTER_REGISTRY[OperationDescription("", op, old_version)]


_alias("Cast", 19, 13)
_alias("Reshape", 19, 14)
_alias("Identity", 19, 16)


class _Gelu(nn.Module):
  def __init__(self, approximate: str = "none"):
    super().__init__()
    self.approximate = approximate

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return nn.functional.gelu(x, approximate=self.approximate)


if OperationDescription("", "Gelu", 20) not in _CONVERTER_REGISTRY:

  @add_converter(operation_type="Gelu", version=20)
  def _gelu(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    approx = node.attributes.get("approximate", "none")
    if isinstance(approx, bytes):
      approx = approx.decode()
    return OperationConverterResult(torch_module=_Gelu(approx), onnx_mapping=onnx_mapping_from_node(node))


def _reduce18(op: str):
  if OperationDescription("", op, 18) in _CONVERTER_REGISTRY:
    return

  @add_converter(operation_type=op, version=18)
  def _conv(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    keepdims = node.attributes.get("keepdims", 1)
    axes = None
    if len(node.input_values) > 1 and node.input_values[1]:
      axes = get_const_value(node.input_values[1], graph).tolist()
    return OperationConverterResult(
      torch_module=OnnxReduceStaticAxes(op, axes, keepdims),
      onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
    )


for _op in ("ReduceMean", "ReduceL2", "ReduceMax", "ReduceSum"):
  _reduce18(_op)
