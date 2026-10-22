"""Prepare openpilot's supercombo ONNX for fine-tuning.

* cast FP16 weights / graph to FP32 (inputs ``img``/``big_img`` stay uint8)
* expose intermediate tensors (stage outputs, final spatial map) as extra outputs ("taps")
* split into a vision graph (images -> vision_outputs + taps) and a policy graph
  (feature history + desire + traffic convention -> plan/desire_state), so the policy can be trained
  from cached vision features.

Graph facts (commit b361e952, checkpoint 6a7d09ad.../100):
  vision trunk: FastViT/MobileOne-style reparameterized convs on concat(img, big_img) (24 ch)
  taps: add_1 (stage 0), add_3 (stage 1), add_9 (stage 2), add_11 (stage 3), gelu_18 (final, post-SE)
  policy: GatherND picks the last 9 of 25 history slots (current hidden + 8 past at 5 Hz);
          ``action_t`` is not used by the graph.
"""

from __future__ import annotations

import base64
import pickle
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

TAPS = {"s4": "add_1", "s8": "add_3", "s16": "add_9", "s32": "add_11", "final": "gelu_18"}


def output_slices(model: onnx.ModelProto) -> dict[str, slice]:
  meta = {p.key: p.value for p in model.metadata_props}
  return pickle.loads(base64.b64decode(meta["output_slices"]))


def to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
  model = onnx.ModelProto.FromString(model.SerializeToString())
  g = model.graph
  for i, init in enumerate(g.initializer):
    if init.data_type == TensorProto.FLOAT16:
      arr = numpy_helper.to_array(init).astype(np.float32)
      g.initializer[i].CopyFrom(numpy_helper.from_array(arr, init.name))
  for vi in list(g.input) + list(g.output) + list(g.value_info):
    if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
      vi.type.tensor_type.elem_type = TensorProto.FLOAT
  for n in g.node:
    if n.op_type == "Cast":
      for a in n.attribute:
        if a.name == "to" and a.i == TensorProto.FLOAT16:
          a.i = TensorProto.FLOAT
    if n.op_type == "Constant":
      for a in n.attribute:
        if a.t.data_type == TensorProto.FLOAT16:
          a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
  # Reshape(allowzero=1) only differs from allowzero=0 if the shape contains 0; onnx2torch lacks it.
  # The exported graph also hard-codes batch=1 in every Reshape target (all are batch-first), so the
  # leading 1 is replaced by 0 (= copy input dim) to make the graph batchable.
  inits = {i.name: i for i in g.initializer}
  for n in g.node:
    if n.op_type == "Reshape" and n.input[1] in inits:
      shape = numpy_helper.to_array(inits[n.input[1]]).copy()
      if 0 not in shape.tolist():
        for a in n.attribute:
          if a.name == "allowzero":
            a.i = 0
        if shape[0] == 1:
          shape[0] = 0
          new = numpy_helper.from_array(shape, n.input[1] + "_batch")
          g.initializer.append(new)
          n.input[1] = new.name
  del g.value_info[:]
  return model


def add_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
  model = onnx.ModelProto.FromString(model.SerializeToString())
  inferred = shape_inference.infer_shapes(model)
  vis = {v.name: v for v in inferred.graph.value_info}
  existing = {o.name for o in model.graph.output}
  for n in names:
    if n in existing:
      continue
    if n in vis:
      model.graph.output.append(vis[n])
    else:
      model.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
  return model


def extract(model: onnx.ModelProto, inputs: list[str], outputs: list[str]) -> onnx.ModelProto:
  inferred = shape_inference.infer_shapes(model)
  e = onnx.utils.Extractor(inferred)
  sub = e.extract_model(inputs, outputs)
  sub.ir_version = model.ir_version
  del sub.opset_import[:]
  sub.opset_import.extend(model.opset_import[:1])
  return sub


def prepare(src: str | Path, out_dir: str | Path) -> dict[str, Path]:
  """Writes ``full_fp32.onnx``, ``vision_fp32.onnx`` and ``policy_fp32.onnx``; returns paths."""
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  m = onnx.load(str(src))
  m32 = to_fp32(m)
  paths = {"full": out_dir / "full_fp32.onnx", "vision": out_dir / "vision_fp32.onnx",
           "policy": out_dir / "policy_fp32.onnx"}
  onnx.save(m32, paths["full"])
  vision = extract(m32, ["img", "big_img"], ["vision_outputs", *TAPS.values()])
  onnx.save(vision, paths["vision"])
  # p_outputs = concat(plan, desire_state, pad[1,2]) - the constant pad is not batchable, so take the
  # two heads directly: p_mul_1 = plan [B,990], p_linear_21 = desire_state [B,8].
  policy = extract(m32, ["fb_full", "dp_out", "tc_out"], ["p_mul_1", "p_linear_21"])
  onnx.save(policy, paths["policy"])
  return paths
