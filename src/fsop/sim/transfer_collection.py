"""Coherent physical-camera expert and model-visited recovery sequences for transfer v2."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fsop.eval.closed_loop import ModelPolicy, mission_brake
from fsop.eval.transfer import transfer_track
from fsop.profile import profile_from_dict
from fsop.sim.env import FSEnv, SimCfg
from fsop.sim.expert import configured_expert
from fsop.sim.randomization import dr_preset
from fsop.sim.recorder import EpisodeWriter


@dataclass(frozen=True)
class InterventionCfg:
  """Collection-only policy. Defaults preserve v2; disabling disagreement permits state visitation."""

  lateral_limit_m: float = 1.25
  steering_disagreement_rad: float | None = .20
  hold_seconds: float = 1.5
  lateral_reference: str = 'centerline'
  lateral_accel_disagreement_mps2: float | None = None

  def __post_init__(self):
    if self.lateral_reference not in ('centerline', 'reference_path'):
      raise ValueError('Unknown collection lateral reference')
    values = [self.lateral_limit_m, self.hold_seconds]
    if self.steering_disagreement_rad is not None:
      values.append(self.steering_disagreement_rad)
    if self.lateral_accel_disagreement_mps2 is not None:
      values.append(self.lateral_accel_disagreement_mps2)
    if any(not np.isfinite(value) or value <= 0 for value in values):
      raise ValueError('Intervention thresholds must be finite and positive, or steering disagreement disabled')


def intervention_lateral_error(obs, mission, config: InterventionCfg):
  """Deviation from the chosen route reference, not a penalty for legitimate racing-line offset."""
  if config.lateral_reference == 'reference_path' and mission.racing_line is not None:
    return obs.lat - mission._at(obs.s, mission.racing_line.offset)
  return obs.lat


def takeover_deadline(now, lateral, steering_gap, previous, config: InterventionCfg, *,
                      speed=None, wheelbase=None, steering=None, expert_steering=None):
  disagreement = (config.steering_disagreement_rad is not None
                  and abs(steering_gap) > config.steering_disagreement_rad)
  if config.lateral_accel_disagreement_mps2 is not None:
    inputs = (speed, wheelbase, steering, expert_steering)
    if any(value is None or not np.isfinite(value) for value in inputs) or wheelbase <= 0:
      raise ValueError('Dynamic collection guard requires finite speed and vehicle geometry')
    # Kinematic curvature proxy, not a claim about actual tire-force capability.
    accel_gap = speed**2 / wheelbase * abs(np.tan(steering) - np.tan(expert_steering))
    disagreement |= accel_gap > config.lateral_accel_disagreement_mps2
  if abs(lateral) > config.lateral_limit_m or disagreement:
    return max(previous, now + config.hold_seconds)
  return previous


def collect(spec: dict, root: str) -> dict:
  root = Path(root)
  out = root / spec['id']
  identity = dict(spec)
  intervention = InterventionCfg(**spec.get('intervention', {}))
  if spec['collector'] == 'dagger':
    identity['checkpoint_sha256'] = hashlib.sha256(Path(spec['checkpoint']).read_bytes()).hexdigest()
  if (out / 'collection.json').exists():
    saved = json.loads((out / 'collection.json').read_text())
    if saved['identity'] != identity:
      raise ValueError('Collection identity changed; use a distinct dataset')
    return saved['row']
  if out.exists():
    raise ValueError(f'Incomplete published episode: {out}')
  scratch = root / '_incomplete' / spec['id']
  if scratch.exists():
    shutil.rmtree(scratch)  # only this incomplete, unpublished attempt
  scratch.mkdir(parents=True)
  profile = profile_from_dict(spec['profile'])
  cfg = SimCfg(physical_camera=True, dr='light', randomize_expert=False,
               random_start=spec.get('random_start', True), max_time=spec['seconds'])
  cfg.noise.enabled = False
  dr = dr_preset('light')
  dr.cam_height_jitter = dr.cam_pitch_deg = dr.cam_roll_deg = dr.cam_yaw_deg = 0.
  started = time.monotonic()
  env = FSEnv(transfer_track(spec['track_seed']), cfg, spec['scene_seed'], profile=profile,
              dr_cfg=dr, expert_params=configured_expert(profile.vehicle))
  try:
    policy = ModelPolicy.from_checkpoint(spec['checkpoint']) if spec['collector'] == 'dagger' else None
    if policy is not None:
      policy.bev = False  # unused by the driving policy and collection labels
    writer = EpisodeWriter(scratch, env, spec['id'], int(spec['seconds'] * 5) + 3, record_every=4)
    obs = env.observe()
    valid, interventions, controls = [], [], []
    desire = np.zeros(8, np.float32)
    takeover_until = 0.
    ticks, expert_ticks = 0, 0
    while not env.done:
      target_steer, target_accel = env.expert_action()
      steer, accel = target_steer, target_accel
      takeover = policy is None
      if policy is not None:
        steer, accel, _ = policy(obs, env)  # always update queues, including expert-controlled ticks
        # Collection-only safety intervention: true track error and true expert action.
        # Hysteresis permits recovery sequences instead of per-frame control switching.
        error = intervention_lateral_error(obs, env.mission, intervention)
        takeover_until = takeover_deadline(obs.t, error, steer - target_steer, takeover_until, intervention,
                                            speed=obs.state.vx, wheelbase=env.vp.wheelbase,
                                            steering=steer, expert_steering=target_steer)
        takeover = obs.t < takeover_until
        if takeover:
          steer, accel = target_steer, target_accel
      accel = mission_brake(accel, obs, env)
      ticks += 1
      expert_ticks += int(takeover)
      desire = np.maximum(desire, obs.desire)
      if obs.frame % 4 == 0:
        plan = env.label()
        valid.append(env.expert.last_rollout.get('contact_samples', 0) == 0)
        interventions.append(takeover)
        controls.append([steer, accel, target_steer, target_accel])
        obs.desire = desire
        writer.add(obs, plan)
        desire = np.zeros(8, np.float32)
      obs = env.step(steer, accel, add_noise=False, render=policy is not None or (obs.frame + 1) % 4 == 0)
    meta = writer.close(env.referee.summary())
    episode = writer.dir
    np.save(episode / 'label_valid.npy', np.array(valid, bool))
    np.save(episode / 'expert_intervention.npy', np.array(interventions, bool))
    np.save(episode / 'controls.npy', np.array(controls, np.float32))
    meta.update(split=spec['split'], configuration=spec['configuration'], collector=spec['collector'],
                valid_label_frames=int(sum(valid)), expert_control_fraction=expert_ticks / ticks,
                collector_privileged_inputs=['true track lateral error', 'expert action', 'mission progress']
                  + (['reference racing-line offset'] if intervention.lateral_reference == 'reference_path' else []),
                intervention=asdict(intervention),
                collection_identity=identity)
    (episode / 'meta.json').write_text(json.dumps(meta, indent=2))
    row = {k: meta[k] for k in ('id', 'track_id', 'discipline', 'frames', 'split', 'collector', 'configuration')}
    row.update(seconds=time.monotonic()-started, valid_frames=int(sum(valid)),
               expert_control_fraction=expert_ticks/ticks, contacts=meta['summary']['doo'])
    (episode / 'collection.json').write_text(json.dumps({'identity': identity, 'row': row}, indent=2))
    episode.rename(out)  # publish only a complete episode
    print(spec['id'], row, flush=True)
    return row
  finally:
    env.renderer.close()
