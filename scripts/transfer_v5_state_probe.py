"""Counterfactual speed information test after matched training, on an existing train configuration."""
import copy
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml
from transfer_diagnostics import environment

from fsop.eval.closed_loop import ModelPolicy
from fsop.sim.expert import T_IDXS


def run():
  root = Path('runs/transfer_state_information_v5')
  root.mkdir(exist_ok=False)
  configuration = yaml.safe_load(Path('configs/distributions/transfer_v2_design.yaml').read_text())['configurations'][3]
  checkpoints = ['runs/transfer_minimal_v1/ckpt/best.pt',
                 'runs/transfer_state_constant_pilot_v5/ckpt/step_000300.pt',
                 'runs/transfer_state_speed_pilot_v5/ckpt/step_000300.pt']
  results = []
  for checkpoint in checkpoints:
    env = environment(configuration, True, 880004, 7104, False, 12.)
    try:
      policy = ModelPolicy.from_checkpoint(checkpoint)
      policy.bev = False
      rng, expert = copy.deepcopy(env.rng.bit_generator.state), copy.deepcopy(env.expert)
      rows, predictions, targets = [], [], []
      for speed in (0., .5, 1., 1.5):
        env.state.vx = speed
        env.rng.bit_generator.state = copy.deepcopy(rng)
        env.expert = copy.deepcopy(expert)
        obs = env.observe()
        policy.reset()
        _, _, info = policy(obs, env)
        target = env.label()
        predictions.append(info['plan'])
        targets.append(target)
        rows.append(dict(speed=speed,
                         image_hashes={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in obs.images.items()},
                         feasible=env.expert.last_rollout['contact_samples'] == 0,
                         target_x_035=float(np.interp(.35, T_IDXS, target[:, 0])),
                         predicted_x_035=float(np.interp(.35, T_IDXS, info['plan'][:, 0]))))
      assert all(r['image_hashes'] == rows[0]['image_hashes'] and r['feasible'] for r in rows)
      result = dict(checkpoint=checkpoint, sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    rows=rows, max_prediction_difference=float(np.max(np.abs(np.array(predictions)-predictions[0]))),
                    mean_abs_x_035_error=float(np.mean([abs(r['predicted_x_035']-r['target_x_035']) for r in rows])))
      if policy.model.state_mode != 'speed':
        assert result['max_prediction_difference'] == 0
      results.append(result)
      print(json.dumps(result), flush=True)
      np.savez_compressed(root/f'{Path(checkpoint).parts[1]}.npz', predicted=predictions, target=targets)
    finally:
      env.renderer.close()
  (root/'report.json').write_text(json.dumps(dict(configuration=configuration, track_seed=880004, scene_seed=7104,
    results=results, limitation='One counterfactual reset state on a training configuration; not a transfer result.'),
    indent=2))
  with tarfile.open(root/'source.tar.gz', 'w:gz') as archive:
    for folder in ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'uv.lock'):
      archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)


if __name__ == '__main__':
  run()
