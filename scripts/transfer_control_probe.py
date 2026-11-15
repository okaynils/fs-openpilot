"""Compare controller-relevant errors on fixed validation observations before training v2."""
import json
from pathlib import Path

import numpy as np

from fsop.data.sim_dataset import Episode, split_episodes
from fsop.eval.controller import PlanPurePursuit
from fsop.eval.open_loop import plan_errors, run_episode_model
from fsop.model.build import load_fs
from fsop.train.device import pick_device


def run(output='runs/transfer_control_probe_v2'):
  root = Path(output)
  if root.exists():
    raise ValueError('Use a distinct probe output name')
  root.mkdir(parents=True)
  device = pick_device()
  rows = []
  dataset = Path('data/sim/transfer_train_v1')
  for name in ('cuda_racing_v6', 'transfer_image_v1', 'transfer_minimal_v1'):
    model = load_fs(f'runs/{name}/ckpt/best.pt').to(device).eval()
    for eid in split_episodes(dataset, 'val'):
      ep = Episode(dataset / eid)
      out = run_episode_model(model, ep, device, bev=False)
      ctrl = PlanPurePursuit(ep.vehicle)
      pred = np.array([ctrl(p, s[4]) for p, s in zip(out['mu'], ep.states, strict=True)])
      target = np.array([ctrl(p, s[4]) for p, s in zip(ep.plan, ep.states, strict=True)])
      mask = np.arange(ep.T) >= 5
      error = np.abs(pred - target)[mask]
      metrics = plan_errors(out['mu'], ep.plan, out['std'])
      rows.append({'checkpoint': name, 'episode': eid, 'frames': int(mask.sum()),
                   'ade_4s': float(metrics['ade_4s'][mask].mean()),
                   'steering_mae_rad': float(error[:, 0].mean()),
                   'acceleration_mae_mps2': float(error[:, 1].mean()),
                   'wrong_turn_fraction': float(((pred[mask, 0] * target[mask, 0] < 0)
                                                 & (np.abs(target[mask, 0]) > .05)).mean()),
                   'origin_offset_m': float(np.linalg.norm(out['mu'][mask, 0, :2], axis=-1).mean()),
                   'near_speed_mae_mps': float(metrics['speed_0.5s'][mask].mean())})
    del model
  result = {'dataset': str(dataset), 'split': 'val', 'rows': rows}
  (root / 'report.json').write_text(json.dumps(result, indent=2))
  for name in dict.fromkeys(r['checkpoint'] for r in rows):
    subset = [r for r in rows if r['checkpoint'] == name]
    print(name, {key: round(float(np.mean([r[key] for r in subset])), 4)
                 for key in ('ade_4s', 'steering_mae_rad', 'wrong_turn_fraction',
                             'acceleration_mae_mps2', 'origin_offset_m')}, flush=True)
  return result


if __name__ == '__main__':
  run()
