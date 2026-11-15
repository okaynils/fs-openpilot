"""Short validation failure trace; no final-test data or checkpoint selection."""
import runpy
from pathlib import Path

import yaml

api = runpy.run_path('scripts/transfer_diagnostics.py')
cfg = yaml.safe_load(Path('configs/eval/transfer_v2_val.yaml').read_text())['configurations'][2]
root = Path('runs/transfer_launch_probe_v2')
if root.exists():
  raise ValueError('Preserve existing diagnostic')
root.mkdir(parents=True)
for name, checkpoint in [('previous', 'runs/transfer_minimal_v1/ckpt/best.pt'),
                         ('expert500', 'runs/transfer_expert_v2/ckpt/step_000500.pt')]:
  api['capture'](cfg, checkpoint, root/name, track_seed=98011, scene_seed=5101, max_seconds=12.)
