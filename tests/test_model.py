from pathlib import Path

import pytest

PREP = Path("artifacts/models/prep")
SRC = Path("artifacts/models/driving_supercombo_b361e952.onnx")


@pytest.mark.skipif(not SRC.exists(), reason="run `fsop model fetch && fsop model prepare` first")
def test_conversion_parity_cpu():
  from fsop.model.verify import verify

  r = verify(SRC, PREP, devices=("cpu",))
  assert r["torch_cpu_vs_fp32"]["plan"]["max_abs"] < 1e-3
  assert r["torch_cpu_vs_fp16"]["plan"]["rel"] < 5e-3


@pytest.mark.skipif(not (PREP / "vision_fp32.onnx").exists(), reason="model not prepared")
def test_fs_model_shapes():
  import torch

  from fsop.model.build import SupercomboFS

  m = SupercomboFS()
  b = {"img": torch.zeros(2, 12, 128, 256), "big_img": torch.zeros(2, 12, 128, 256),
       "history": torch.zeros(2, 24, 512), "desire": torch.zeros(2, 25, 8), "traffic": torch.zeros(2, 2)}
  o = m.forward_images(b)
  assert o["plan_mu"].shape == (2, 33, 15)
  assert o["bev"]["heat"].shape == (2, 4, 64, 64)
  assert o["img"]["wide"]["heat"].shape == (2, 4, 64, 128)
