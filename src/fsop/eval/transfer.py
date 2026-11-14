"""Paired transfer metrics and deployment-runtime temporal parity."""
from __future__ import annotations

import numpy as np


def trajectory_clearance(plan, cones, vehicle, cone_radius=.15):
  """Signed cone-to-rectangular-footprint clearance for each predicted pose.

  Negative values denote overlap; cone radius is conservative and shared across
  models. This metric does not replace the simulator's swept contact referee.
  """
  if len(cones) == 0:
    return np.full(len(plan), np.inf)
  delta = np.asarray(cones)[None, :, :2] - plan[:, None, :2]
  yaw = plan[:, 11, None]
  x = np.cos(yaw) * delta[..., 0] + np.sin(yaw) * delta[..., 1]
  y = -np.sin(yaw) * delta[..., 0] + np.cos(yaw) * delta[..., 1]
  front, rear = vehicle.lf + vehicle.front_overhang, vehicle.lr + vehicle.rear_overhang
  qx = np.abs(x - (front - rear) / 2) - (front + rear) / 2
  qy = np.abs(y) - vehicle.width / 2
  signed = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
  return (signed - cone_radius).min(1)


def temporal_parity(checkpoint, exported, episode, frames=12):
  """Exercise deployment queues at 20 Hz with explicitly repeated 5 Hz recordings."""
  from fsop.geometry.camera import CameraMount
  from fsop.geometry.yuv import model_to_rgb
  from fsop.profile import TeamProfile
  from fsop.runtime import FSDriver

  profile = TeamProfile(mount=CameraMount(**episode.meta.get('calibrated_mount', episode.meta['mount'])),
                        vehicle=episode.vehicle)
  ref, got = FSDriver(checkpoint, profile=profile), FSDriver(exported, profile=profile)
  errors = []
  for k in range(min(episode.T, frames)):
    images = {c: model_to_rgb(episode.frames[c][k]) for c in ('road', 'wide')}
    for _ in range(4):
      a = ref.step(images, float(episode.states[k, 4]), already_model_frame=True)
      b = got.step(images, float(episode.states[k, 4]), already_model_frame=True)
      errors.append([np.abs(a.plan - b.plan).max(), abs(a.steer - b.steer), abs(a.accel - b.accel),
                     np.abs(a.extra['raw']['hidden_state'] - b.extra['raw']['hidden_state']).max()])
  array = np.asarray(errors)
  result = dict(zip(('plan_max_abs', 'steer_max_abs', 'accel_max_abs', 'hidden_max_abs'),
                    array.max(0).tolist(), strict=True))
  result['ticks'] = len(array)
  result['input_cadence'] = '5 Hz recorded images repeated four times; runtime queues at 20 Hz'
  if not np.isfinite(array).all() or result['plan_max_abs'] > .001 or result['hidden_max_abs'] > 1e-5:
    raise ValueError(f'Temporal deployment parity failed: {result}')
  return result


def paired_open_loop(dataset, checkpoints, output, split="all"):
  import hashlib
  import json
  from pathlib import Path

  from fsop.data.sim_dataset import Episode, load_index
  from fsop.data.targets import world_to_ego
  from fsop.eval.open_loop import plan_errors, run_episode_model
  from fsop.model.build import load_fs
  from fsop.sim.expert import T_IDXS
  from fsop.train.device import pick_device

  device = pick_device()
  rows = []
  for checkpoint in checkpoints:
    model = load_fs(checkpoint).to(device).eval()
    for item in load_index(dataset):
      if split != 'all' and item.get('split') != split:
        continue
      ep = Episode(Path(dataset) / item['id'])
      predicted = run_episode_model(model, ep, device, bev=False)
      errors = plan_errors(predicted['mu'], ep.plan, predicted['std'])
      clearance, expert_clearance = [], []
      for k in range(5, ep.T):
        cones = world_to_ego(ep.cones_xy, *ep.pose(k))
        mask = T_IDXS <= 4
        clearance.append(trajectory_clearance(predicted['mu'][k, mask], cones, ep.vehicle).min())
        expert_clearance.append(trajectory_clearance(ep.plan[k, mask], cones, ep.vehicle).min())
      rows.append({'checkpoint': str(checkpoint), 'episode': item['id'],
                   'configuration': item.get('configuration', ep.meta.get('distribution', {}).get('seed', 'legacy')),
                   'regime': item.get('regime'),
                   **{k: float(v[5:].mean()) for k, v in errors.items()},
                   'minimum_predicted_clearance_m': float(np.min(clearance)),
                   'predicted_contact_frame_rate': float(np.mean(np.asarray(clearance) < 0)),
                   'expert_contact_frame_rate': float(np.mean(np.asarray(expert_clearance) < 0))})
    del model
  report = {'checkpoint_sha256': {str(c): hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in checkpoints},
            'dataset_manifest_sha256': hashlib.sha256((Path(dataset) / 'manifest.json').read_bytes()).hexdigest(),
            'rows': rows}
  Path(output).parent.mkdir(parents=True, exist_ok=True)
  Path(output).write_text(json.dumps(report, indent=2))
  return report


def context_ablation(dataset, checkpoint, output):
  """Validation-only intervention: replace calibrated context groups with default values."""
  import json
  from pathlib import Path

  from fsop.data.sim_dataset import Episode, split_episodes
  from fsop.eval.open_loop import plan_errors, run_episode_model
  from fsop.geometry.camera import CameraMount
  from fsop.model.build import load_fs
  from fsop.model.conditioning import context_fields, context_scale, geometry_context
  from fsop.sim.vehicle import VehicleParams
  from fsop.train.device import pick_device

  device = pick_device()
  model = load_fs(checkpoint).to(device).eval()
  if model.context_adapter is None:
    raise ValueError('Context ablation requires a conditioned checkpoint')
  version = model.context_version
  fields = context_fields(version)
  default = geometry_context(CameraMount(), VehicleParams(), version)
  groups = {'default_translation': fields[:3],
            'default_orientation': ('camera_roll', 'camera_pitch', 'camera_yaw'),
            'default_camera': tuple(f for f in fields if f.startswith('camera_')),
            'default_vehicle': tuple(f for f in fields if not f.startswith('camera_')),
            'default_footprint': ('wheelbase', 'width', 'front_extent', 'rear_extent'),
            'default_speed_limit': ('max_speed',),
            'default_dynamic_limits': ('max_steer', 'max_accel', 'max_brake'),
            'default_response': ('tau_steer', 'tau_accel')}
  rows = []
  for eid in split_episodes(dataset, 'val'):
    ep = Episode(Path(dataset) / eid)
    context = geometry_context(CameraMount(**ep.meta.get('calibrated_mount', ep.meta['mount'])), ep.vehicle, version)
    hidden = None
    interventions = [('calibrated', [])]
    interventions += [(name, [fields.index(f) for f in names if f in fields]) for name, names in groups.items()
                      if any(f in fields for f in names)]
    interventions += [('calibration_plus', []), ('calibration_minus', [])]
    for name, indices in interventions:
      altered = context.copy()
      altered[indices] = default[indices]
      if name.startswith('calibration_'):
        sign = 1 if name == 'calibration_plus' else -1
        altered[:3] += sign * .005 / context_scale(version)[:3]
        if version == 1:
          altered[3:6] += sign * np.deg2rad(.25) / context_scale(version)[3:6]
      out = run_episode_model(model, ep, device, bev=False, geometry_override=altered,
                              encoded_hidden=hidden, return_hidden=True)
      hidden = out['encoded_hidden']
      metrics = plan_errors(out['mu'], ep.plan, out['std'])
      rows.append({'episode': eid, 'intervention': name,
                   **{k: float(v[5:].mean()) for k, v in metrics.items()}})
  result = {'checkpoint': str(checkpoint), 'split': 'val', 'rows': rows}
  Path(output).write_text(json.dumps(result, indent=2))
  return result


def transfer_profile(configuration):
  from fsop.profile import load_profile

  profile = load_profile(f"configs/profiles/generic_{configuration['family']}.yaml")
  profile.name = configuration['name']
  for key in ('x', 'y', 'z'):
    setattr(profile.mount, key, configuration[key])
  for key in ('roll', 'pitch', 'yaw'):
    setattr(profile.mount, key, float(np.deg2rad(configuration[f'{key}_deg'])))
  v = profile.vehicle
  v.front_overhang = max(v.front_overhang, profile.mount.x - v.lf + .03)
  v.rear_overhang = max(v.rear_overhang, -profile.mount.x - v.lr + .03)
  v.length = v.wheelbase + v.front_overhang + v.rear_overhang
  for camera in profile.cameras:
    camera.width, camera.height = 640, 360
    camera.fx = camera.fy = 350.
    camera.cx, camera.cy = 320., 180.
  return profile


def transfer_track(seed):
  from fsop.track.generate import TrackGenCfg, generate_track

  # Short, full-width autocross circuits let a low-speed rover complete the mission.
  return generate_track(seed, TrackGenCfg(discipline='autocross', method='fourier',
                                         length_min=80, length_max=120, slalom_prob=0))


def closed_loop_case(configuration, checkpoints, track_seed, scene_seed, recovery=False, max_seconds=100.):
  import copy
  import time

  from fsop.data.targets import visible_cones, world_to_ego
  from fsop.eval.closed_loop import ModelPolicy, RuntimePolicy
  from fsop.geometry.camera import CAMERA_MODELS, CameraModel, project_world
  from fsop.geometry.mounting import body_image_mask, body_occluded
  from fsop.sim.drive import expert_policy
  from fsop.sim.env import FSEnv, SimCfg
  from fsop.sim.randomization import dr_preset
  from fsop.sim.render.panda3d_backend import Panda3DRenderer

  profile = transfer_profile(configuration)
  track = transfer_track(track_seed)
  if recovery:
    x, y, heading = track.start_pose
    track.start_pose = (x - .5 * np.sin(heading), y + .5 * np.cos(heading), heading + .08)
  cameras = {c.name: CameraModel(c.name, c.K, c.width, c.height) for c in profile.cameras}
  renderer = Panda3DRenderer(cameras)
  try:
    results = []
    for checkpoint in [*checkpoints, 'expert']:
      started = time.time()
      config = SimCfg(physical_camera=True, dr='light', max_time=max_seconds, randomize_expert=False)
      config.noise.enabled = False
      dr = dr_preset('light')
      dr.cam_height_jitter = dr.cam_pitch_deg = dr.cam_roll_deg = dr.cam_yaw_deg = 0.
      from fsop.sim.expert import configured_expert
      env = FSEnv(copy.deepcopy(track), config, scene_seed, renderer=renderer, profile=profile, dr_cfg=dr,
                  expert_params=configured_expert(profile.vehicle))
      if checkpoint == 'expert':
        policy = expert_policy
      elif str(checkpoint).endswith('.onnx'):
        policy = RuntimePolicy(checkpoint)
      else:
        policy = ModelPolicy.from_checkpoint(checkpoint)
      obs = env.observe()
      rows, tracking, prediction, latency = [], [], [], []
      pending = {}
      body_fraction = {k: float(body_image_mask(c, env.T_ego_cam(), env.vp).mean())
                       for k, c in CAMERA_MODELS.items()}
      initial_progress = env.s
      while not env.done:
        policy_started = time.perf_counter()
        steer, accel, info = policy(obs, env)
        latency.append(time.perf_counter() - policy_started)
        cones = world_to_ego(env.cones_xy, obs.state.x, obs.state.y, obs.state.yaw)
        stationary = np.zeros((1, 15))
        clearance = trajectory_clearance(stationary, cones, env.vp)[0]
        visible = visible_cones(cones, env.T_ego_cam())
        points = np.c_[cones, np.zeros(len(cones))]
        sensor_visible = np.zeros(len(cones), bool)
        from fsop.geometry.warp import _cam_rot
        for camera in profile.cameras:
          offset = np.eye(4)
          offset[:3, :3] = _cam_rot(*np.deg2rad([camera.roll_deg, camera.pitch_deg, camera.yaw_deg]))
          sensor_visible |= project_world(points, np.eye(4), env.T_ego_cam() @ offset,
                                          cameras[camera.name])[2]
        visible &= sensor_visible & ~env.referee.down & ~body_occluded(points, env.T_ego_cam(), env.vp)
        rows.append([obs.t, abs(obs.lat), obs.state.steer, obs.state.r, clearance, visible.sum(), obs.state.vx])
        if obs.frame in pending:
          tracking.append(float(np.linalg.norm(np.array([obs.state.x, obs.state.y]) - pending.pop(obs.frame))))
        if 'plan' in info and obs.frame % 20 == 0:
          from fsop.sim.expert import T_IDXS
          plan = info['plan']
          mask = T_IDXS <= 4
          expert_plan = env.label()
          prediction.append(float(np.linalg.norm(plan[mask, :2] - expert_plan[mask, :2], axis=1).mean()))
          xy = np.array([np.interp(1., T_IDXS, plan[:, i]) for i in (0, 1)])
          c, s = np.cos(obs.state.yaw), np.sin(obs.state.yaw)
          pending[obs.frame + 20] = np.array([obs.state.x, obs.state.y]) + np.array([[c, -s], [s, c]]) @ xy
        obs = env.step(steer, accel, add_noise=False)
      values = np.asarray(rows)
      summary = env.referee.summary()
      recovery_window = values[(values[:, 0] >= 4) & (values[:, 0] <= 6), 1]
      summary.update(checkpoint=str(checkpoint), configuration=configuration['name'], regime=configuration['regime'],
                     track_seed=track_seed, scene_seed=scene_seed, recovery=recovery,
                     completion_fraction=float(np.clip((env.s - initial_progress) /
                                               (env.mission.s_end - initial_progress), 0, 1)),
                     minimum_clearance_m=float(values[:, 4].min()),
                     rms_lateral_m=float(np.sqrt(np.mean(values[:, 1] ** 2))),
                     rms_steer_rate_rad_s=float(np.sqrt(np.mean((np.diff(values[:, 2]) / env.dt) ** 2))),
                     max_yaw_rate_rad_s=float(np.abs(values[:, 3]).max()),
                     steering_saturation_fraction=float(np.mean(np.abs(values[:, 2]) >= .95 * env.vp.max_steer)),
                     mean_visible_cones=float(values[:, 5].mean()), body_occlusion_fraction=body_fraction,
                     recovery_lateral_4_to_6s_m=float(recovery_window.mean()) if len(recovery_window) else None,
                     online_expert_ade_4s=float(np.mean(prediction)) if prediction else None,
                     one_second_tracking_error_m=float(np.mean(tracking)) if tracking else None,
                     elapsed_seconds=time.time() - started,
                     renderer=renderer.base.win.getGsg().getDriverRenderer(),
                     runtime_physical_sensor_input=str(checkpoint).endswith('.onnx'),
                     input_rate_hz=env.cfg.fps,
                     policy_first_call_ms=latency[0] * 1000,
                     policy_steady_p95_ms=float(np.percentile(latency[min(20, len(latency)-1):], 95) * 1000),
                     privileged_inputs=['simulator wheel speed', 'route desire', 'mission progress for launch/stop']
                       + (['full racing line', 'true vehicle state and dynamics', 'cone map for recovery']
                          if checkpoint == 'expert' else []),
                     expert_privileged=checkpoint == 'expert')
      results.append(summary)
      print(f"{configuration['name']} {checkpoint}: {summary['result']} "
            f"progress={summary['completion_fraction']:.3f} contacts={summary['doo']}", flush=True)
    return results
  finally:
    renderer.close()

