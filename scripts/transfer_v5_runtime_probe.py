"""Export both trained pilot contracts and compare fresh physical-camera sequences."""
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml
from transfer_diagnostics import environment

from fsop.eval.closed_loop import ModelPolicy, RuntimePolicy
from fsop.model.export import export_onnx
from fsop.train.device import strict_inference_precision


def run():
  strict_inference_precision()
  root = Path('runs/transfer_state_runtime_v5')
  root.mkdir(exist_ok=False)
  configurations = yaml.safe_load(Path('configs/eval/transfer_v2_val.yaml').read_text())['configurations'][2:]
  results = []
  for mode in ('constant', 'speed'):
    checkpoint = f'runs/transfer_state_{mode}_pilot_v5/ckpt/step_000300.pt'
    exported = export_onnx(checkpoint, root/f'{mode}.onnx')
    for i, configuration in enumerate(configurations):
      env = environment(configuration, True, 98011, 5101, bool(i), 12.)
      try:
        reference, runtime = ModelPolicy.from_checkpoint(checkpoint), RuntimePolicy(exported)
        obs, errors = env.observe(), []
        for _ in range(120):
          a, b = reference(obs, env), runtime(obs, env)
          errors.append([np.max(np.abs(a[2]['plan']-b[2]['plan'])), abs(a[0]-b[0]), abs(a[1]-b[1])])
          steer, accel = env.expert_action()
          obs = env.step(steer, accel, add_noise=False)
        values = np.asarray(errors)
        row = dict(mode=mode, configuration=configuration['name'], recovery=bool(i), ticks=120,
                   plan_max_abs=float(values[:, 0].max()), steer_max_abs=float(values[:, 1].max()),
                   accel_max_abs=float(values[:, 2].max()),
                   checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                   export_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
                   sequence='Fresh identical 20 Hz physical-camera observations on expert-driven validation states')
        assert np.isfinite(values).all() and row['plan_max_abs'] < .001 and row['steer_max_abs'] < 1e-4, row
        results.append(row)
        print(json.dumps(row), flush=True)
      finally:
        env.renderer.close()
  (root/'report.json').write_text(json.dumps(results, indent=2))
  with tarfile.open(root/'source.tar.gz', 'w:gz') as archive:
    for folder in ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'uv.lock'):
      archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)


if __name__ == '__main__':
  run()
