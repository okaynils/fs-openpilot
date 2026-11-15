"""Measure trajectory-derived steering bias on validation-only model-visited states."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import yaml
from transfer_diagnostics import environment

from fsop.eval.closed_loop import ModelPolicy
from fsop.eval.controller import PlanPurePursuit
from fsop.sim.expert import T_IDXS


def run(checkpoint='runs/transfer_selected_v2/selected.pt', output='runs/transfer_plan_probe_v3'):
  root = Path(output)
  if root.exists():
    raise ValueError('Preserve existing probe')
  root.mkdir(parents=True)
  cfg = yaml.safe_load(Path('configs/eval/transfer_v2_val.yaml').read_text())['configurations'][2]
  env = environment(cfg, True, 98011, 5101, False, 12.)
  try:
    policy = ModelPolicy.from_checkpoint(checkpoint)
    follower = PlanPurePursuit(env.vp)
    obs = env.observe()
    rows = []
    plans, targets = [], []
    while not env.done:
      steer, accel, info = policy(obs, env)
      if obs.frame % 4 == 0:
        target = env.label()
        oracle_steer, oracle_accel = follower(target, obs.state.vx)
        predicted = info['plan']
        rows.append(dict(t=obs.t, lateral=obs.lat, speed=obs.state.vx, steer=steer,
                         oracle_steer=oracle_steer, oracle_accel=oracle_accel,
                         valid_target=env.expert.last_rollout.get('contact_samples', 0) == 0,
                         y_error_035=float(np.interp(.35,T_IDXS,predicted[:,1]-target[:,1])),
                         x_error_035=float(np.interp(.35,T_IDXS,predicted[:,0]-target[:,0]))))
        plans.append(predicted)
        targets.append(target)
      obs = env.step(steer, accel, add_noise=False)
    report = dict(configuration=cfg, checkpoint=checkpoint,
                  checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  tail_mean_abs_lateral=float(np.mean([abs(r['lateral']) for r in rows if r['t'] >= 10.])),
                  track_seed=98011, scene_seed=5101, rows=rows,
                  summary=env.referee.summary())
    (root/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(root/'plans.npz', predicted=plans, target=targets)
    for start in (0, 4, 8):
      window=[r for r in rows if start<=r['t']<start+4]
      print(start,{key:float(np.mean([r[key] for r in window])) for key in
                   ('steer','oracle_steer','y_error_035','x_error_035','valid_target')},flush=True)
    with tarfile.open(root/'source.tar.gz','w:gz') as archive:
      for folder in ('src','scripts','configs','tests','pyproject.toml','uv.lock'):
        archive.add(folder,filter=lambda item:None if '__pycache__' in item.name else item)
  finally:
    env.renderer.close()


if __name__ == '__main__':
  run(*sys.argv[1:])
