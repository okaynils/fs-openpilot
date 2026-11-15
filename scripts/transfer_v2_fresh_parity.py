"""Compare deployment and training inference on identical fresh physical-camera sequences."""
import json
from pathlib import Path

import numpy as np
import yaml
from transfer_diagnostics import environment

from fsop.eval.closed_loop import ModelPolicy, RuntimePolicy


def verify():
  suite = yaml.safe_load(Path('configs/eval/transfer_v2_runtime.yaml').read_text())
  results = []
  for i, configuration in enumerate(suite['configurations']):
    env = environment(configuration, True, suite['track_seeds'][0], suite['scene_seeds'][0], bool(i), 100.)
    try:
      torch_policy = ModelPolicy.from_checkpoint('runs/transfer_selected_v2/selected.pt')
      runtime = RuntimePolicy('runs/transfer_selected_v2/candidate.onnx')
      obs = env.observe()
      errors = []
      for _ in range(120):
        a, b = torch_policy(obs, env), runtime(obs, env)
        errors.append([np.max(np.abs(a[2]['plan']-b[2]['plan'])), abs(a[0]-b[0]), abs(a[1]-b[1])])
        steer, accel = env.expert_action()
        obs = env.step(steer, accel, add_noise=False)
      values = np.asarray(errors)
      row = dict(configuration=configuration['name'], ticks=len(values), recovery=bool(i),
                 plan_max_abs=float(values[:, 0].max()), steer_max_abs=float(values[:, 1].max()),
                 accel_max_abs=float(values[:, 2].max()),
                 sequence='identical fresh 20 Hz physical sensor images; expert-driven states')
      results.append(row)
      print(row, flush=True)
      if not np.isfinite(values).all() or row['plan_max_abs'] > .001:
        raise ValueError(f'Fresh physical-camera parity failed: {row}')
    finally:
      env.renderer.close()
  Path('runs/transfer_selected_v2/fresh_parity.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
  verify()
