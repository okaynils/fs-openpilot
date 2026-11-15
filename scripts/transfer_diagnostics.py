"""Fixed, post-selection probes separating camera/policy errors from trajectory tracking."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fsop.eval.closed_loop import ModelPolicy, mission_brake
from fsop.eval.controller import PlanPurePursuit
from fsop.eval.transfer import transfer_profile, transfer_track
from fsop.sim.env import FSEnv, SimCfg
from fsop.sim.expert import T_IDXS, configured_expert
from fsop.sim.randomization import dr_preset


def environment(configuration, render, track_seed=99001, scene_seed=4101, recovery=False, max_seconds=100.):
  profile = transfer_profile(configuration)
  cfg = SimCfg(render=render, physical_camera=render, dr='light', max_time=max_seconds, randomize_expert=False)
  cfg.noise.enabled = False
  dr = dr_preset('light')
  dr.cam_height_jitter = dr.cam_pitch_deg = dr.cam_roll_deg = dr.cam_yaw_deg = 0
  track = transfer_track(track_seed)
  if recovery:
    x, y, heading = track.start_pose
    track.start_pose = (x - .5 * np.sin(heading), y + .5 * np.cos(heading), heading + .08)
  return FSEnv(track, cfg, scene_seed, profile=profile, dr_cfg=dr,
               expert_params=configured_expert(profile.vehicle))


def oracle_tracking(configuration, output, track_seed=99001, scene_seed=4101, recovery=False, max_seconds=100.):
  results = []
  for mode in ('expert', 'oracle_pp'):
    env = environment(configuration, False, track_seed, scene_seed, recovery, max_seconds)
    controller = PlanPurePursuit(env.vp)
    obs = env.observe()
    rows = []
    while not env.done:
      if mode == 'expert':
        steer, accel = env.expert_action()
      else:
        steer, accel = controller(env.label(), obs.state.vx)
        accel = mission_brake(accel, obs, env)
      rows.append([obs.t, obs.lat, obs.state.vx, steer, accel])
      obs = env.step(steer, accel, add_noise=False, render=False)
    results.append({'mode': mode, 'summary': env.referee.summary(), 'trace': rows})
  Path(output).write_text(json.dumps({'configuration': configuration, 'track_seed': track_seed,
                                    'scene_seed': scene_seed, 'recovery': recovery,
                                    'max_seconds': max_seconds, 'rendering': False,
                                    'privileged': 'full expert plan or expert action', 'results': results}, indent=2))
  return results


def capture(configuration, checkpoint, output, track_seed=99001, scene_seed=4101,
            recovery=False, max_seconds=100.):
  import cv2

  from fsop.geometry.camera import CAMERA_MODELS, project_world
  from fsop.geometry.mounting import body_occluded

  output = Path(output)
  env = environment(configuration, True, track_seed, scene_seed, recovery, max_seconds)
  policy = ModelPolicy.from_checkpoint(checkpoint)
  obs = env.observe()
  frames, trace = [], []
  while not env.done:
    steer, accel, info = policy(obs, env)
    if obs.frame % 20 == 0:
      predicted, expert = info['plan'], env.label()
      trace.append({'time': obs.t, 'x': obs.state.x, 'y': obs.state.y, 'speed': obs.state.vx,
                    'lateral': obs.lat, 'steer': steer, 'accel': accel, 'contacts': int(env.referee.down.sum()),
                    'ade_4s': float(np.linalg.norm(predicted[T_IDXS <= 4, :2] - expert[T_IDXS <= 4, :2],
                                                   axis=1).mean()),
                    'predicted_speed': float(np.interp(.3, T_IDXS, predicted[:, 3])),
                    'expert_speed': float(np.interp(.3, T_IDXS, expert[:, 3]))})
      if obs.frame // 20 in (0, 1, 2, 4, 6, 8, 10, 12):
        panels = []
        for slot, camera in CAMERA_MODELS.items():
          image = obs.images[slot].copy()
          for plan, color in ((expert, (30, 230, 60)), (predicted, (255, 50, 70))):
            points = np.c_[plan[:, :2], np.zeros(len(plan))]
            pixels, _, visible = project_world(points, np.eye(4), env.T_ego_cam(), camera)
            visible &= ~body_occluded(points, env.T_ego_cam(), env.vp)
            for pixel in pixels[visible]:
              cv2.circle(image, tuple(pixel.astype(int)), 3, color, -1)
          panels.append(image)
        image = np.hstack(panels)
        cv2.putText(image, f"t={obs.t:.1f}s v={obs.state.vx:.1f} contacts={int(env.referee.down.sum())} "
                    "red=prediction green=expert", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        frames.append(cv2.resize(image, (768, 192)))
    obs = env.step(steer, accel, add_noise=False)
  report = {'configuration': configuration, 'checkpoint': checkpoint, 'track_seed': track_seed,
            'scene_seed': scene_seed, 'recovery': recovery, 'max_seconds': max_seconds,
            'summary': env.referee.summary(), 'trace': trace,
            'renderer': env.renderer.base.win.getGsg().getDriverRenderer()}
  output.with_suffix('.json').write_text(json.dumps(report, indent=2))
  cv2.imwrite(str(output.with_suffix('.jpg')), np.vstack(frames)[..., ::-1])
  env.renderer.close()
  print(output.name, report['summary'], flush=True)
  return report


def run(output='runs/transfer_diagnostics_v1'):
  root = Path(output)
  if root.exists():
    raise ValueError('Use a fresh diagnostic directory')
  root.mkdir(parents=True)
  configs = yaml.safe_load(Path('configs/eval/transfer_v1.yaml').read_text())['configurations']
  rear = next(c for c in configs if c['name'] == 'formula_high_rear')
  normal = {**rear, 'name': 'formula_reference_camera', 'x': 0., 'y': 0., 'z': 1.1,
            'roll_deg': 0., 'pitch_deg': 0., 'yaw_deg': 0.}
  oracle_tracking(rear, root / 'oracle_tracking.json')
  capture(normal, 'runs/cuda_racing_v6/ckpt/best.pt', root / 'reference_baseline')
  capture(rear, 'runs/cuda_racing_v6/ckpt/best.pt', root / 'rear_baseline')
  capture(rear, 'runs/transfer_minimal_v1/ckpt/best.pt', root / 'rear_candidate')


if __name__ == '__main__':
  run()
