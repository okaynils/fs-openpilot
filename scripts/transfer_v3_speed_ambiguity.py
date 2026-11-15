"""Counterfactual reset-state diagnostic: identical pixels/geometry, different measured speeds."""
import copy
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml
from transfer_diagnostics import environment

from fsop.eval.closed_loop import ModelPolicy
from fsop.model.conditioning import geometry_context
from fsop.sim.expert import T_IDXS

root=Path('runs/transfer_speed_ambiguity_v3')
if root.exists():
  raise ValueError('Preserve existing diagnostic')
root.mkdir(parents=True)
config=yaml.safe_load(Path('configs/distributions/transfer_v2_design.yaml').read_text())['configurations'][3]
checkpoint='runs/transfer_selected_v2/selected.pt'
env=environment(config,True,880004,7104,False,12.)
try:
  policy=ModelPolicy.from_checkpoint(checkpoint)
  policy.bev=False
  rng=copy.deepcopy(env.rng.bit_generator.state)
  expert=copy.deepcopy(env.expert)
  results=[]
  predictions=[]
  targets=[]
  for speed in (0.,.5,1.,1.5):
    assert speed<=env.vp.max_speed
    env.state.vx=speed
    env.rng.bit_generator.state=copy.deepcopy(rng)
    env.expert=copy.deepcopy(expert)
    obs=env.observe()
    policy.reset()
    _,_,info=policy(obs,env)
    target=env.label()
    predictions.append(info['plan'])
    targets.append(target)
    results.append(dict(speed=speed,
                        image_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in obs.images.items()},
                        geometry=geometry_context(env.calibrated_mount(),env.vp,2).tolist(),
                        desire=obs.desire.tolist(),T_world_ego=obs.T_world_ego.tolist(),
                        feasible=env.expert.last_rollout['contact_samples']==0,
                        target_x_035=float(np.interp(.35,T_IDXS,target[:,0])),
                        predicted_x_035=float(np.interp(.35,T_IDXS,info['plan'][:,0]))))
  report=dict(configuration=config,track_seed=880004,scene_seed=7104,checkpoint=checkpoint,
              checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),results=results,
              identical_policy_inputs=all(all(r[k]==results[0][k] for k in
                   ('image_hashes','geometry','desire','T_world_ego')) for r in results),
              max_prediction_difference=float(np.max(np.abs(np.array(predictions)-predictions[0]))),
              max_target_xy_difference=float(np.max(np.abs(np.array(targets)[:,:,:2]-targets[0][:,:2]))),
              limitation='Reset/empty-history ambiguity only. With useful history visual motion may reveal speed. '
                         'This does not establish every driving failure or justify training by itself.')
  (root/'report.json').write_text(json.dumps(report,indent=2))
  np.savez_compressed(root/'plans.npz',predicted=predictions,target=targets)
  with tarfile.open(root/'source.tar.gz','w:gz') as archive:
    for folder in ('src','configs','scripts','tests','pyproject.toml','uv.lock'):
      archive.add(folder,filter=lambda item:None if '__pycache__' in item.name else item)
  print(json.dumps({k:v for k,v in report.items() if k not in ('configuration','results')},indent=2))
  print(json.dumps([{k:v for k,v in r.items() if k in ('speed','feasible','target_x_035','predicted_x_035')}
                    for r in results],indent=2))
finally:
  env.renderer.close()
