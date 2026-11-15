"""Retain the original steering safeguard after unguarded reference collection failed."""
import json
import tarfile
from pathlib import Path

import yaml

from fsop.sim.transfer_collection import collect

root=Path('data/sim/transfer_guarded_collection_probe_v3')
if root.exists():
  raise ValueError('Preserve existing diagnostic')
previous=Path('data/sim/transfer_reference_train_v3')
old=json.loads((previous/'generation.json').read_text())
frames,valid=old['frames'],old['valid_frames']
if valid/frames >= .95:
  raise ValueError('Guarded follow-up requires the predeclared unguarded quality gate to fail')
policy=yaml.safe_load(Path('configs/distributions/transfer_collection_guarded_v3.yaml').read_text())
specs=[{**s,'intervention':policy['intervention'],'collection_policy':policy,'split':'diagnostic'}
       for s in json.loads((previous/'manifest.json').read_text())['specs']
       if s['id'] in ('train_v2_00_dagger','train_v2_03_dagger','train_v2_13_dagger')]
assert len(specs)==3
root.mkdir(parents=True)
results=[collect(s,str(root)) for s in specs]
rover=[r for r in results if r['id']!='train_v2_00_dagger']
report=dict(specs=specs,results=results,previous=[r for r in old['rows'] if r['id'] in {s['id'] for s in specs}],
            gate=dict(valid_at_least_95_percent=
                      sum(r['valid_frames'] for r in results)/sum(r['frames'] for r in results)>=.95,
                      rover_expert_fraction_below_75_percent=
                      sum(r['expert_control_fraction'] for r in rover)/len(rover)<.75,
                      contacts_below_three=sum(r['contacts'] for r in results)<3),diagnostic_only=True)
(root/'report.json').write_text(json.dumps(report,indent=2))
with tarfile.open(root/'source.tar.gz','w:gz') as archive:
  for folder in ('src','configs','scripts','tests','pyproject.toml','uv.lock'):
    archive.add(folder,filter=lambda item:None if '__pycache__' in item.name else item)
print(json.dumps(report,indent=2))
