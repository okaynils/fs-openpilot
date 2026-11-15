"""Align replacement episode order with its control and audit actual new observations."""
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import yaml

from fsop.data.sim_dataset import Episode

root = Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/sim/transfer_reference_train_v3')
source = Path('data/sim/transfer_train_v2')
index = root/'index.json'
original = json.loads((source/'index.json').read_text())['episodes']
rows = json.loads(index.read_text())['episodes']
by_id = {r['id']:r for r in rows}
assert len(by_id) == len(rows) == len(original)
assert set(by_id) == {r['id'] for r in original}
manifest = json.loads((root/'manifest.json').read_text())
new_ids = {s['id'] for s in manifest['specs']}
pre_hash = hashlib.sha256(index.read_bytes()).hexdigest()
# Match the original order as well as sampler seed. Replacement collection can finish
# out of order, and all retained episodes must keep their original sampling positions.
index.write_text(json.dumps({'episodes':[by_id[r['id']] for r in original]},indent=2))
families = {c['name']:c['family'] for c in yaml.safe_load(
    Path('configs/distributions/transfer_v2_design.yaml').read_text())['configurations']}
audit = []
for row in original:
  if row['id'] not in new_ids:
    continue
  ep = Episode(root/row['id'])
  old = Episode(source/row['id'])
  audit.append(dict(id=row['id'], family=families[row['configuration']],
                    frames=ep.T, valid_frames=int(ep.label_valid.sum()),
                    expert_control_fraction=ep.meta['expert_control_fraction'],
                    old_expert_control_fraction=old.meta['expert_control_fraction'],
                    max_pose_step=float(np.max(np.abs(np.diff(ep.camera_poses,axis=0)))),
                    max_origin_error=float(np.max(np.abs(ep.plan[:,0,:3]))),
                    contacts=ep.meta['summary']['doo'],recorded_contacts=int(np.isfinite(ep.knock_times).sum())))
rovers=[r for r in audit if r['family']=='rover']
report=dict(rows=audit,pre_alignment_index_sha256=pre_hash,index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
            order_matches_source=True,
            valid_fraction=sum(r['valid_frames'] for r in audit)/sum(r['frames'] for r in audit),
            rover_expert_fraction=float(np.mean([r['expert_control_fraction'] for r in rovers])),
            old_rover_expert_fraction=float(np.mean([r['old_expert_control_fraction'] for r in rovers])))
report['audit_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(root/'recovery_audit_source.py').write_bytes(Path(__file__).read_bytes())
report['gates']=dict(valid_fraction_at_least_95_percent=report['valid_fraction']>=.95,
                     lower_rover_expert_fraction=report['rover_expert_fraction']<report['old_rover_expert_fraction'],
                     vehicle_origin=all(r['max_origin_error']<1e-5 for r in audit),
                     contacts_match=all(r['contacts']==r['recorded_contacts'] for r in audit),
                     coherent_motion=all(r['max_pose_step']<.05 for r in audit))
(root/'recovery_audit.json').write_text(json.dumps(report,indent=2))
# Restrict preview to the new collection, without changing the training index.
preview=root/'recovery_preview'
preview.mkdir(exist_ok=True)
for eid in sorted(new_ids):
  dst=preview/eid
  if not dst.exists():
    dst.symlink_to((root/eid).resolve(),target_is_directory=True)
(preview/'index.json').write_text(json.dumps({'episodes':[by_id[r['id']] for r in original if r['id'] in new_ids]}))
sys.argv=['scripts/transfer_v2_preview.py',str(preview)]
runpy.run_path(sys.argv[0],run_name='__main__')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
