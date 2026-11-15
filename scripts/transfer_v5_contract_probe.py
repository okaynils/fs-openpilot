"""Exercise real Torch/ONNX state-input contracts before any speed-conditioned fit."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from fsop.model.build import SupercomboFS
from fsop.model.export import export_onnx
from fsop.runtime import FSDriver
from fsop.train.device import strict_inference_precision


def run():
  strict_inference_precision()
  root = Path('runs/transfer_state_contract_v5')
  root.mkdir(exist_ok=False)
  source = torch.load('runs/transfer_minimal_v1/ckpt/best.pt', map_location='cpu', weights_only=False)
  results = []
  for mode in ('speed', 'constant'):
    torch.manual_seed(184193)
    model = SupercomboFS(conditioned=True, anchored_context=True, context_version=2, state_mode=mode,
                         image_head=any(k.startswith('img.') for k in source['model'])).eval()
    missing, unexpected = model.load_state_dict(source['model'], strict=False)
    assert missing and all(k.startswith('motion_adapter.') for k in missing) and not unexpected
    # Nonzero synthetic weights test actual input plumbing; these are not trained candidates.
    torch.nn.init.normal_(model.motion_adapter[-1].weight, std=.005)
    contract = dict(source['input_contract'], state_mode=mode, state_version=1, speed_scale_mps=15.)
    checkpoint = root/f'{mode}.pt'
    torch.save(dict(model=model.state_dict(), input_contract=contract, inference_only=True), checkpoint)
    artifact = export_onnx(checkpoint, root/f'{mode}.onnx')
    torch_driver = FSDriver(checkpoint, 'configs/profiles/generic_rover.yaml')
    onnx_driver = FSDriver(artifact, 'configs/profiles/generic_rover.yaml', providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(184193)
    frames = {cam: rng.integers(30, 220, (256, 512, 3), dtype=np.uint8) for cam in ('road', 'wide')}
    errors = []
    for speed in (0., .5, 1., 1.5, 1., .5, 0., 0.):
      a = torch_driver.step(frames, speed, already_model_frame=True)
      b = onnx_driver.step(frames, speed, already_model_frame=True)
      error = float(np.max(np.abs(a.plan-b.plan)))
      assert error < .002, (mode, error)
      assert abs(a.steer-b.steer) < 1e-4
      errors.append(error)
    if mode == 'speed':
      for bad in (float('nan'), float('inf'), None):
        for driver in (torch_driver, onnx_driver):
          before = copy.deepcopy(driver.ts.feat_q)
          try:
            driver.step(frames, bad, already_model_frame=True)
          except ValueError:
            assert all(np.array_equal(a, b) for a, b in zip(before, driver.ts.feat_q, strict=True))
          else:
            raise AssertionError('Invalid speed accepted')
    results.append(dict(mode=mode, steps=8, max_plan_error=max(errors), synthetic_weights=True))
  (root/'report.json').write_text(json.dumps(results, indent=2))
  print(json.dumps(results), flush=True)


if __name__ == '__main__':
  run()
