"""Identical-prefix, current-feature norm intervention; no weight or runtime changes."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from transfer_diagnostics import environment

from fsop.eval.closed_loop import ModelPolicy
from fsop.geometry.camera import CAMERA_MODELS, project_world
from fsop.geometry.mounting import body_occluded


def instrument(original, state):
  def call(features, desire, traffic):
    state['norm'] = float(features[:, -1].norm())
    if state['active']:
      current = torch.nn.functional.normalize(features[:, -1:], dim=-1)
      features = torch.cat([features[:, :-1], current], dim=1)
    return original(features, desire, traffic)
  return call


def run(checkpoint, output):
  root = Path(output)
  root.mkdir(exist_ok=False)
  configuration = yaml.safe_load(Path('configs/eval/transfer_v2_val.yaml').read_text())['configurations'][2]
  results = []
  for mode in ('unchanged', 'normalize_after_12s'):
    env = environment(configuration, True, 98011, 5101, False, 60.)
    try:
      policy = ModelPolicy.from_checkpoint(checkpoint)
      policy.bev = False
      state = dict(active=False, norm=None)
      policy.model.base.policy_raw = instrument(policy.model.base.policy_raw, state)
      obs, rows, panels = env.observe(), [], []
      target = None
      while not env.done:
        state['active'] = mode == 'normalize_after_12s' and obs.frame >= 240
        steer, accel, info = policy(obs, env)
        reference = env.mission._at(obs.s, env.mission.racing_line.offset)
        if obs.frame % 4 == 0:
          target = env.label()  # diagnostic only, never used to select an action
        rows.append(dict(t=obs.t, x=obs.state.x, y=obs.state.y, yaw=obs.state.yaw, speed=obs.state.vx,
                         steer=steer, accel=accel, progress=obs.s, reference_error=obs.lat-reference,
                         norm_before=state['norm'], intervened=state['active'],
                         events=[e.kind for e in obs.events]))
        if obs.frame in (0, 160, 240, 320, 480, 800, 1120):
          image = obs.images['wide'].copy()
          for plan, color in ((target, (30, 230, 60)), (info['plan'], (255, 50, 70))):
            points = np.c_[plan[:, :2], np.zeros(len(plan))]
            uv, _, valid = project_world(points, np.eye(4), env.T_ego_cam(), CAMERA_MODELS['wide'])
            valid &= ~body_occluded(points, env.T_ego_cam(), env.vp)
            for pixel in uv[valid]:
              cv2.circle(image, tuple(pixel.astype(int)), 3, color, -1)
          label = f"{mode} t={obs.t:.0f} err={obs.lat-reference:.2f} norm={state['norm']:.2f}"
          cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
          panels.append(image)
        obs = env.step(steer, accel, add_noise=False)
      record = dict(mode=mode, rows=rows, summary=env.referee.summary(), final_progress=env.s,
                    final_reference_error=rows[-1]['reference_error'])
      results.append(record)
      (root/f'{mode}.json').write_text(json.dumps(record, indent=2))
      cv2.imwrite(str(root/f'{mode}.jpg'), np.vstack(panels)[..., ::-1])
      print(mode, record['summary'], 'final reference error', record['final_reference_error'], flush=True)
    finally:
      env.renderer.close()
  keys = ('x', 'y', 'yaw', 'speed', 'steer', 'accel', 'progress', 'reference_error', 'norm_before')
  prefix = [np.array([[r[k] for k in keys] for r in result['rows'] if r['t'] < 12.-1e-7]) for result in results]
  assert prefix[0].shape == prefix[1].shape
  prefix_error = float(np.max(np.abs(prefix[0]-prefix[1])))
  assert prefix_error < 1e-6, prefix_error
  report = dict(checkpoint=checkpoint, checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                configuration=configuration, track_seed=98011, scene_seed=5101, seconds=60, switch_frame=240,
                prefix_max_error=prefix_error, results=results,
                intervention='Only the current adapted feature is normalized to unit length after frame 240. '
                             'Raw visual history, weights, geometry, measured speed and controller are unchanged.',
                privileged='Both: exact simulator speed, route desire and mission launch/stop assistance. '
                           'Expert labels are visualization only and do not control the vehicle.')
  (root/'report.json').write_text(json.dumps(report, indent=2))
  with tarfile.open(root/'source.tar.gz', 'w:gz') as archive:
    for folder in ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'uv.lock'):
      archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)
  print('prefix_max_error', prefix_error, flush=True)


if __name__ == '__main__':
  run(*sys.argv[1:])
