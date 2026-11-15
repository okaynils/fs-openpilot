"""Bounded, validation-only test of spatial lookahead with a fixed transfer checkpoint."""
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml
from transfer_diagnostics import environment

from fsop.eval.closed_loop import ModelPolicy
from fsop.eval.controller import PlanPurePursuit
from fsop.sim.expert import T_IDXS


class SpatialProbeFollower(PlanPurePursuit):
  """Experimental lateral target; longitudinal commands exactly match the legacy follower."""

  def __call__(self, plan, v):
    _, accel = super().__call__(plan, v)
    points = np.asarray(plan)[T_IDXS <= 4, :2].copy()
    points[:, 0] += self.vp.lr
    points = points[np.isfinite(points).all(1) & (points[:, 0] > 0)]
    if not len(points):
      return 0., accel
    distances = np.linalg.norm(points, axis=1)
    lookahead = max(2 * self.vp.wheelbase, .8 * max(v, 0.))
    crossing = np.flatnonzero(distances >= lookahead)
    if len(crossing):
      idx = int(crossing[0])
      if idx:
        ratio = (lookahead-distances[idx-1]) / max(distances[idx]-distances[idx-1], 1e-8)
        point = points[idx-1] + ratio * (points[idx]-points[idx-1])
      else:
        point = points[idx]
    else:
      point = points[int(np.argmax(distances))]
    ld2 = max(float(point @ point), (self.vp.wheelbase / 1.55)**2)
    steer = np.arctan(2 * self.vp.wheelbase * point[1] / ld2)
    return float(np.clip(steer, -self.vp.max_steer, self.vp.max_steer)), accel


def run():
  root = Path('runs/transfer_spatial_probe_v3')
  if root.exists():
    raise ValueError('Preserve existing probe')
  root.mkdir(parents=True)
  checkpoint = Path('runs/transfer_selected_v2/selected.pt')
  configuration = yaml.safe_load(Path('configs/eval/transfer_v2_val.yaml').read_text())['configurations'][2]
  results = []
  for mode, follower in [('legacy', PlanPurePursuit), ('spatial', SpatialProbeFollower)]:
    env = environment(configuration, True, 98011, 5101, False, 12.)
    try:
      policy = ModelPolicy.from_checkpoint(checkpoint)
      policy.ctrl = follower(env.vp)
      obs = env.observe()
      trace = []
      while not env.done:
        steer, accel, info = policy(obs, env)
        expert_steer, expert_accel = env.expert_action()
        t = np.clip(.35 + .03*obs.state.vx, .35, 1.)
        x, y = [float(np.interp(t, T_IDXS, info['plan'][:, axis])) for axis in (0, 1)]
        trace.append(dict(t=obs.t, lateral=obs.lat, speed=obs.state.vx, steer=steer, accel=accel,
                          expert_steer=expert_steer, expert_accel=expert_accel,
                          legacy_target_x=x, legacy_target_y=y))
        obs = env.step(steer, accel, add_noise=False)
      result = dict(mode=mode, summary=env.referee.summary(), trace=trace,
                    max_abs_lateral=max(abs(r['lateral']) for r in trace),
                    tail_mean_abs_lateral=float(np.mean([abs(r['lateral']) for r in trace if r['t'] >= 10.])))
      results.append(result)
      print({k: v for k, v in result.items() if k != 'trace'}, flush=True)
    finally:
      env.renderer.close()
  report = dict(configuration=configuration, track_seed=98011, scene_seed=5101, max_seconds=12.,
                checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                privileged=['simulator speed', 'route desire', 'mission progress launch/stop',
                            'expert commands for diagnostics only'], results=results,
                reproduces_drift=results[0]['tail_mean_abs_lateral'] > 1.)
  (root/'report.json').write_text(json.dumps(report, indent=2))
  with tarfile.open(root/'source.tar.gz', 'w:gz') as archive:
    for folder in ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'uv.lock'):
      archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)
  if not report['reproduces_drift']:
    raise AssertionError('Short baseline probe did not reproduce the known drift; do not infer a fix')


if __name__ == '__main__':
  run()
