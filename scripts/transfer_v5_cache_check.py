"""Read-only proof that preserved baseline visual caches also match the v1 initializer."""
import hashlib
import json
from pathlib import Path

import torch

from fsop.data.feature_cache import _done
from fsop.data.sim_dataset import Episode, load_index

baseline = Path('runs/cuda_racing_v6/ckpt/best.pt')
initial = Path('runs/transfer_minimal_v1/ckpt/best.pt')
a, b = (torch.load(path, map_location='cpu', weights_only=False)['model'] for path in (baseline, initial))
keys = [k for k in a if k.startswith('base.vision.')]
assert keys and set(keys) == {k for k in b if k.startswith('base.vision.')}
assert all(torch.equal(a[k], b[k]) for k in keys), 'Different vision requires a distinct feature cache'
expected = hashlib.sha256(baseline.read_bytes()).hexdigest()+':fp32_no_tf32_v1'
dataset = Path('data/sim/transfer_guarded_train_v3')
episodes = load_index(dataset)
for row in episodes:
  episode = Episode(dataset/row['id'])
  folder = episode.cache_dir('transfer_baseline_strict_v2_full')
  assert _done(episode, 'transfer_baseline_strict_v2_full', False)
  assert json.loads((folder/'identity.json').read_text())['vision_identity'] == expected
out = Path('runs/transfer_state_cache_check_v5')
out.mkdir(exist_ok=False)
report = dict(exact_vision_tensor_equality=True, tensors=len(keys), checked_episodes=len(episodes),
              baseline_sha256=expected.split(':')[0], initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
              cache_tag='transfer_baseline_strict_v2_full', cache_modified=False)
(out/'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
