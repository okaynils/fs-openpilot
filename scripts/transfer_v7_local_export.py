"""CPU deployment export/parity while Modal is unavailable; no driving-success claim."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fsop.model.export import export_onnx
from fsop.profile import load_profile
from fsop.runtime import FSDriver, TorchEngine


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
  torch.set_num_threads(2)
  checkpoint = Path('artifacts/experiments/transfer_refresh_current_pilot_v7/step_000300.pt')
  root = Path('artifacts/export/transfer_refresh_current_pilot_v7')
  root.mkdir(exist_ok=False)
  exported = export_onnx(checkpoint, root/'candidate.onnx')
  rows = []
  for family in ('formula', 'rover'):
    profile = load_profile(f'configs/profiles/generic_{family}.yaml')
    reference = FSDriver(exported, profile, providers=['CPUExecutionProvider'])
    reference.engine = TorchEngine(checkpoint, device='cpu')
    runtime = FSDriver(exported, profile, providers=['CPUExecutionProvider'])
    errors = []
    for tick in range(40):
      images = {}
      for camera in profile.cameras:
        x, y = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
        images[camera.name] = np.stack(((x+tick)%256, (y+tick)%256, (x+y+tick)%256), -1).astype(np.uint8)
      speed = min(2., tick/20.)
      a, b = reference.step(images, speed), runtime.step(images, speed)
      errors.append([float(np.abs(a.plan-b.plan).max()), abs(a.steer-b.steer), abs(a.accel-b.accel)])
    values = np.array(errors)
    row = dict(profile=profile.name, frames=40, plan_max_abs=float(values[:, 0].max()),
               steering_max_abs=float(values[:, 1].max()), acceleration_max_abs=float(values[:, 2].max()))
    assert np.isfinite(values).all() and row['plan_max_abs'] < .001 and row['steering_max_abs'] < 1e-4, row
    rows.append(row)
    print(json.dumps(row), flush=True)
  report = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                export=str(exported), export_sha256=digest(exported), results=rows,
                source_sha256=digest(__file__),
                limitation='CPU Torch/ONNX parity on synthetic raw-camera ramps through actual FSDriver warping, '
                           'speed input, independent temporal queues and controller. These are not physically '
                           'rendered driving observations and establish neither transfer nor driving performance.')
  (root/'parity.json').write_text(json.dumps(report, indent=2))
  (root/'source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
  run()
