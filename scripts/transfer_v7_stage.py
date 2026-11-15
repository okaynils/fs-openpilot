"""Stage equal-count paired correction datasets without modifying source episodes."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

OLD = Path('data/sim/transfer_guarded_train_v3')
NEW = Path('data/sim/transfer_refresh_guarded_pilot_v7')
CACHE = 'transfer_baseline_strict_v2_full'


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_masks(left, right):
  """Match supervised timestamps while retaining all observations in each sequence."""
  n = min(len(left), len(right))
  common = left[:n] & right[:n]
  masks = [np.zeros_like(left, dtype=bool), np.zeros_like(right, dtype=bool)]
  for mask in masks:
    mask[:n] = common
  return masks


def run():
  quality = json.loads((NEW/'inspection.json').read_text())['quality']
  if not all(quality.values()):
    raise ValueError(f'Collection quality gate failed: {quality}')
  checkpoint = Path('runs/transfer_state_speed_v5/ckpt/step_000250.pt')
  baseline = Path('runs/cuda_racing_v6/ckpt/best.pt')
  states = [torch.load(p, map_location='cpu', weights_only=False)['model'] for p in (baseline, checkpoint)]
  keys = {k for k in states[0] if k.startswith('base.vision.')}
  if not keys or keys != {k for k in states[1] if k.startswith('base.vision.')}:
    raise ValueError('Vision cache contract mismatch')
  if not all(torch.equal(states[0][k], states[1][k]) for k in keys):
    raise ValueError('Cannot reuse baseline visual features with changed vision tensors')
  vision_contract = dict(tensors_compared=len(keys), identical=True,
                         baseline_sha256=digest(baseline), initialization_sha256=digest(checkpoint))
  del states
  rows = json.loads((OLD/'index.json').read_text())['episodes']
  fresh = json.loads((NEW/'index.json').read_text())['episodes']
  configs = sorted({r['configuration'] for r in fresh})
  masks = {}
  for config in configs:
    ids = [f'{config}_{arm}' for arm in ('previous', 'current')]
    original = [np.load(NEW/eid/'label_valid.npy') for eid in ids]
    for eid, mask in zip(ids, paired_masks(*original), strict=True):
      masks[eid] = mask
  for arm in ('previous', 'current'):
    root = Path(f'data/sim/transfer_refresh_{arm}_train_v7')
    root.mkdir(exist_ok=False)
    selected = [r for r in fresh if r['id'].endswith('_'+arm)]
    manifest = dict(schema=1, arm=arm, vision_contract=vision_contract, old_manifest_sha256=digest(OLD/'manifest.json'),
                    refresh_manifest_sha256=digest(NEW/'manifest.json'),
                    sampling='Matched valid frame indices per configuration; all observation/history frames retained.',
                    old_metadata_sha256={r['id']: digest(OLD/r['id']/'meta.json') for r in rows}, refresh=[])
    for row in rows:
      shutil.copytree(OLD/row['id'], root/row['id'],
                      ignore=lambda directory, names: {n for n in names
                                                       if n.startswith('cache_') and n != f'cache_{CACHE}'})
      assert digest(root/row['id']/'meta.json') == manifest['old_metadata_sha256'][row['id']]
    for row in selected:
      eid = row['id']
      shutil.copytree(NEW/eid, root/eid, ignore=shutil.ignore_patterns('cache_*'))
      np.save(root/eid/'label_valid.npy', masks[eid])
      manifest['refresh'].append(dict(id=eid, source_metadata_sha256=digest(NEW/eid/'meta.json'),
                                     original_mask_sha256=digest(NEW/eid/'label_valid.npy'),
                                     derived_mask_sha256=digest(root/eid/'label_valid.npy'),
                                     supervised_frames=int(masks[eid].sum())))
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (root/'index.json').write_text(json.dumps(dict(episodes=rows+selected), indent=2))
    (root/'staging_source.py').write_bytes(Path(__file__).read_bytes())
    print(root, sum(r['supervised_frames'] for r in manifest['refresh']), flush=True)


if __name__ == '__main__':
  run()
