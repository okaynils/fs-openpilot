"""Read-only audit of actual sampled speed/history coverage; no image or final-test reads."""
import hashlib
import json
from pathlib import Path

import numpy as np

from fsop.data.sim_dataset import SimDataset


def run():
  root = Path('runs/transfer_state_coverage_v5')
  root.mkdir(exist_ok=False)
  dataset_root = Path('data/sim/transfer_guarded_train_v3')
  manifest_before = hashlib.sha256((dataset_root/'manifest.json').read_bytes()).hexdigest()
  dataset = SimDataset(dataset_root, 'train', context_version=2)
  rows, grouped = [], {}
  for ei, ep in enumerate(dataset.eps):
    ks = np.array([k for e, k in dataset.index if e == ei], dtype=int)
    speed = ep.states[ks, 4]
    # Explicit geometry bins avoid guessing a platform identity from names.
    family = 'wheelbase_below_1m' if ep.vehicle.wheelbase < 1. else 'wheelbase_at_least_1m'
    summary = dict(samples=len(ks), empty_history=int((ks == 0).sum()),
                   partial_history=int(((ks > 0) & (ks < 8)).sum()),
                   speed_le_0_1=int((speed <= .1).sum()), speed_0_1_to_0_5=int(((speed > .1) & (speed <= .5)).sum()),
                   speed_0_5_to_1_5=int(((speed > .5) & (speed <= 1.5)).sum()),
                   speed_gt_1_5=int((speed > 1.5).sum()))
    bucket = grouped.setdefault(family, {k: 0 for k in summary})
    for key, value in summary.items():
      bucket[key] += value
    rows.append(dict(episode=ep.path.name, geometry_bin=family, **summary,
                     first_recorded_speed=float(ep.states[0, 4]), first_label_valid=bool(ep.label_valid[0]),
                     sampled_speed_min=float(speed.min()), sampled_speed_max=float(speed.max())))
  assert manifest_before == hashlib.sha256((dataset_root/'manifest.json').read_bytes()).hexdigest()
  report = dict(dataset=str(dataset_root), manifest_sha256=manifest_before, episodes=len(dataset.eps),
                samples=len(dataset), geometry_bins=grouped, rows=rows, dataset_modified=False,
                sample_rule='SimDataset default min_k=1; valid labels only. k=0 would have empty visual history.',
                limitation='Counts describe coverage, not the causal explanation for driving failures.')
  (root/'report.json').write_text(json.dumps(report, indent=2))
  (root/'source.py').write_bytes(Path(__file__).read_bytes())
  print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
  run()
