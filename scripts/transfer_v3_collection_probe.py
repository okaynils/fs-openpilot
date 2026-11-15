"""Paired training-configuration diagnostic; these recordings are not training or validation data."""
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml

from fsop.sim.transfer_collection import collect


def run():
  root = Path('data/sim/transfer_collection_probe_v3')
  if root.exists():
    raise ValueError('Preserve the collection probe')
  root.mkdir(parents=True)
  configuration = yaml.safe_load(Path('configs/distributions/transfer_v2_design.yaml').read_text())['configurations'][3]
  policy = yaml.safe_load(Path('configs/distributions/transfer_collection_v3.yaml').read_text())
  specs, rows = [], []
  for name, intervention in [('legacy', {}), ('lateral_only', policy['intervention'])]:
    spec = dict(id=name, configuration=configuration['name'], profile=configuration['profile'],
                split='diagnostic', collector='dagger', checkpoint='runs/transfer_minimal_v1/ckpt/best.pt',
                track_seed=880004, scene_seed=7104, seconds=24., random_start=True,
                intervention=intervention, collection_policy='v2_default' if name=='legacy' else policy)
    specs.append(spec)
    result = collect(spec, str(root))
    states = np.load(root/name/'states.npy')
    # Recorded state columns: time, world pose, body velocity; use counterfactual plan validity separately.
    result['mean_speed'] = float(states[:, 4].mean())
    result['valid_fraction'] = result['valid_frames']/result['frames']
    rows.append(result)
  report = dict(specs=specs, results=rows, diagnostic_only=True,
                gate=dict(expert_fraction_below_75_percent=rows[1]['expert_control_fraction'] < .75,
                          feasible_fraction_at_least_95_percent=rows[1]['valid_fraction'] >= .95,
                          contacts_below_three=rows[1]['contacts'] < 3))
  (root/'report.json').write_text(json.dumps(report,indent=2))
  with tarfile.open(root/'source.tar.gz','w:gz') as archive:
    for folder in ('src','configs','scripts','tests','pyproject.toml','uv.lock'):
      archive.add(folder,filter=lambda item:None if '__pycache__' in item.name else item)
  print(json.dumps(report['results'],indent=2),flush=True)


if __name__ == '__main__':
  run()
