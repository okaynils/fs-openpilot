"""Paired precision diagnostic on fixed validation samples; never opens final data."""
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

from fsop.data.sim_dataset import SimDataset, collate
from fsop.model.build import load_fs
from fsop.train.device import autocast_ctx, strict_inference_precision
from fsop.train.losses import follower_steering, plan_nll


@torch.no_grad()
def run():
  strict_inference_precision()
  root = Path('runs/transfer_precision_probe_v4')
  root.mkdir(exist_ok=False)
  dataset = SimDataset('data/sim/transfer_guarded_train_v3', 'val', 'images',
                       context_version=2, history_mode='fresh', image_targets_on=False)
  indices = sorted(np.random.default_rng(184192).choice(len(dataset), 64, replace=False).tolist())
  checkpoints = ['runs/transfer_minimal_v1/ckpt/best.pt',
                 'runs/transfer_vision_frozen_pilot_v4/ckpt/step_000100.pt',
                 'runs/transfer_vision_last_pilot_v4/ckpt/step_000100.pt']
  results = []
  for checkpoint in checkpoints:
    model = load_fs(checkpoint).cuda().eval()
    rows = []
    for start in range(0, len(indices), 8):
      selected = indices[start:start+8]
      batch = {k: v.cuda() for k, v in collate([dataset[i] for i in selected]).items()}
      outputs = {}
      for precision in ('fp32', 'bf16'):
        with autocast_ctx(torch.device('cuda'), enabled=precision == 'bf16'):
          out = model.forward_images(batch, heads=('plan',))
        mu, std = out['plan_mu'].float(), out['plan_logstd'].float()
        outputs[precision] = (mu, std, follower_steering(mu, batch['speed'], batch['control_vehicle']))
      expected = follower_steering(batch['plan'], batch['speed'], batch['control_vehicle'])
      for j, index in enumerate(selected):
        row = dict(index=index, episode=dataset.eps[dataset.index[index][0]].path.name,
                   frame=dataset.index[index][1], speed=float(batch['speed'][j]),
                   target_steer=float(expected[j]))
        for precision, (mu, std, steer) in outputs.items():
          row[precision] = dict(steer=float(steer[j]),
                                nll=float(plan_nll(mu[j:j+1], std[j:j+1], batch['plan'][j:j+1])),
                                near_xy=mu[j, :8, :2].cpu().tolist())
        rows.append(row)
    delta = np.array([r['bf16']['steer'] - r['fp32']['steer'] for r in rows])
    result = dict(checkpoint=checkpoint, sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  mean_abs_steering_delta_rad=float(np.abs(delta).mean()),
                  max_abs_steering_delta_rad=float(np.abs(delta).max()), rows=rows)
    for precision in ('fp32', 'bf16'):
      result[precision] = dict(mean_abs_steering_error_rad=float(np.mean([
        abs(r[precision]['steer'] - r['target_steer']) for r in rows])),
        mean_nll=float(np.mean([r[precision]['nll'] for r in rows])))
    results.append(result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    del model
  (root/'report.json').write_text(json.dumps(dict(seed=184192, indices=indices, results=results,
    limitation='Identical recorded validation inputs; no closed-loop causal claim. Current-weight '
               'history is FP32 in both arms, exactly as in training.'), indent=2))
  with tarfile.open(root/'source.tar.gz', 'w:gz') as archive:
    for folder in ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'uv.lock'):
      archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)


if __name__ == '__main__':
  run()
