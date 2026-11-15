"""Change only the lateral-error reference after the paired centerline diagnostic failed."""
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml

from fsop.sim.transfer_collection import collect


def run():
  previous = Path('data/sim/transfer_collection_probe_v3')
  old = json.loads((previous/'report.json').read_text())
  root = Path('data/sim/transfer_reference_collection_probe_v3')
  if root.exists():
    raise ValueError('Preserve the reference collection probe')
  root.mkdir(parents=True)
  policy = yaml.safe_load(Path('configs/distributions/transfer_collection_reference_v3.yaml').read_text())
  spec = {**old['specs'][1], 'id': 'reference_path', 'intervention': policy['intervention'],
          'collection_policy': policy}
  result = collect(spec, str(root))
  states = np.load(root/spec['id']/'states.npy')
  before = np.load(previous/'lateral_only'/'states.npy')
  n = min(len(states),len(before))
  result.update(mean_speed=float(states[:,4].mean()), valid_fraction=result['valid_frames']/result['frames'],
                mean_pose_displacement_from_reference_controlled_run_m=float(
                    np.linalg.norm(states[:n,1:3]-before[:n,1:3],axis=1).mean()))
  report = dict(spec=spec, previous_report=str(previous/'report.json'), previous=old['results'][1], result=result,
                diagnostic_only=True, gate=dict(expert_fraction_below_75_percent=result['expert_control_fraction']<.75,
                                               feasible_fraction_at_least_95_percent=result['valid_fraction']>=.95,
                                               contacts_below_three=result['contacts']<3))
  (root/'report.json').write_text(json.dumps(report,indent=2))
  with tarfile.open(root/'source.tar.gz','w:gz') as archive:
    for folder in ('src','configs','scripts','tests','pyproject.toml','uv.lock'):
      archive.add(folder,filter=lambda item:None if '__pycache__' in item.name else item)
  print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
  run()
