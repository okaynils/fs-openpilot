from __future__ import annotations

import os

import torch


def pick_device(pref: str = "auto") -> torch.device:
  if pref != "auto":
    return torch.device(pref)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return torch.device("mps")
  return torch.device("cpu")


def autocast_ctx(device: torch.device, enabled: bool = True):
  if device.type == "cuda" and enabled:
    return torch.autocast("cuda", dtype=torch.bfloat16)
  import contextlib

  return contextlib.nullcontext()


def strict_inference_precision() -> None:
  """Use IEEE FP32 for inference/cache parity with the exported CPU reference.

  cuDNN TF32 convolutions measurably perturb this model's metric trajectory head.
  Explicit BF16 training autocast remains a separate, intentional choice.
  """
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
