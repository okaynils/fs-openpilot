"""Read-only audit of adapter outputs at the frozen policy boundary."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fsop.data.sim_dataset import SimDataset, collate
from fsop.model.build import load_fs


@torch.no_grad()
def run():
  torch.set_num_threads(2)
  root = Path('runs/transfer_feature_norm_audit')
  root.mkdir(exist_ok=False)
  dataset = SimDataset('data/sim/transfer_guarded_train_v3', 'val', cache_tag='transfer_baseline_strict_v2_full',
                       context_version=2)
  indices = sorted(np.random.default_rng(184194).choice(len(dataset), 64, replace=False).tolist())
  batches = [collate([dataset[i] for i in indices[start:start+8]]) for start in range(0, len(indices), 8)]
  checkpoints = ['runs/cuda_racing_v6/ckpt/best.pt', 'runs/transfer_minimal_v1/ckpt/best.pt',
                 'runs/transfer_state_constant_v5/ckpt/step_000250.pt',
                 'runs/transfer_state_speed_v5/ckpt/step_000250.pt',
                 'runs/transfer_state_speed_v5/ckpt/step_001000.pt']
  results = []
  for checkpoint in checkpoints:
    model = load_fs(checkpoint).eval()
    captured = []

    def capture(features, desire, traffic, captured=captured):
      captured.append(features[:, -1].clone())
      return features.new_zeros((len(features), 990)), features.new_zeros((len(features), 8))

    model.base.policy_raw = capture
    raw, adapted = [], []
    for batch in batches:
      model.plan(batch['history'], batch['hidden'], batch['desire'], batch['traffic'],
                 batch['geometry'], batch['speed'])
      raw.append(batch['hidden'])
      adapted.append(captured.pop())
    raw, adapted = torch.cat(raw), torch.cat(adapted)
    ratio = adapted.norm(dim=-1)/raw.norm(dim=-1)
    result = dict(checkpoint=checkpoint, sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  raw_norm_min=float(raw.norm(dim=-1).min()), raw_norm_max=float(raw.norm(dim=-1).max()),
                  adapted_norm_min=float(adapted.norm(dim=-1).min()),
                  adapted_norm_max=float(adapted.norm(dim=-1).max()),
                  norm_ratio_median=float(ratio.median()), norm_ratio_max=float(ratio.max()),
                  correction_norm_mean=float((adapted-raw).norm(dim=-1).mean()),
                  cosine_mean=float(torch.nn.functional.cosine_similarity(raw, adapted).mean()))
    results.append(result)
    print(json.dumps(result), flush=True)
    del model
  (root/'report.json').write_text(json.dumps(dict(seed=184194, indices=indices, results=results,
    limitation='Recorded validation features only; norm changes do not prove a driving failure cause.'), indent=2))
  (root/'source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
  run()
