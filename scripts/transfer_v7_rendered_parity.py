"""Local CPU backend parity on fresh physical-camera validation observations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transfer_diagnostics import environment

from fsop.eval.transfer import transfer_profile
from fsop.runtime import FSDriver, TorchEngine


def run():
  torch.set_num_threads(2)
  checkpoint = Path('artifacts/experiments/transfer_refresh_current_pilot_v7/step_000300.pt')
  exported = Path('artifacts/export/transfer_refresh_current_pilot_v7/candidate.onnx')
  output = Path('benchmarks/transfer_v7/local_rendered_parity.json')
  if output.exists():
    raise ValueError('Preserve the previous parity result')
  configurations = yaml.safe_load(Path('configs/eval/transfer_v2_val.yaml').read_text())['configurations'][2:]
  rows = []
  for i, configuration in enumerate(configurations):
    profile = transfer_profile(configuration)
    reference = FSDriver(exported, profile, providers=['CPUExecutionProvider'])
    reference.engine = TorchEngine(checkpoint, device='cpu')
    runtime = FSDriver(exported, profile, providers=['CPUExecutionProvider'])
    env = environment(configuration, True, 98011, 5101, bool(i), 12.)
    try:
      obs, errors = env.observe(), []
      for _ in range(120):
        a = reference.step(obs.images, obs.state.vx, obs.desire, already_model_frame=True)
        b = runtime.step(obs.images, obs.state.vx, obs.desire, already_model_frame=True)
        errors.append([float(np.abs(a.plan-b.plan).max()), abs(a.steer-b.steer), abs(a.accel-b.accel)])
        steer, accel = env.expert_action()
        obs = env.step(steer, accel, add_noise=False)
      values = np.array(errors)
      row = dict(configuration=configuration['name'], recovery=bool(i), frames=120,
                  plan_max_abs=float(values[:, 0].max()), steering_max_abs=float(values[:, 1].max()),
                  acceleration_max_abs=float(values[:, 2].max()))
      assert np.isfinite(values).all() and row['plan_max_abs'] < .001 and row['steering_max_abs'] < 1e-4, row
      rows.append(row)
      print(json.dumps(row), flush=True)
    finally:
      env.renderer.close()
  report = dict(results=rows, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                export_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                track_seed=98011, scene_seed=5101, backend='Torch CPU versus ONNX CPU',
                limitation='Fresh 20 Hz physical-camera simulator observations, warped by FSEnv; independent '
                           'FSDriver queues. Expert drives every state; exact speed/desire are privileged. '
                           'This verifies numerical temporal parity, not learned-policy driving success or GPU parity.')
  output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
  run()
