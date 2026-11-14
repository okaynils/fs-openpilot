"""Versioned, deterministic episode configuration sampling for transfer experiments."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from fsop.geometry.camera import CameraMount
from fsop.geometry.mounting import MountDynamics
from fsop.profile import load_profile


def sample_profile(distribution: str | Path, seed: int, family: str | None = None):
  path = Path(distribution)
  cfg = yaml.safe_load(path.read_text())
  if cfg['version'] != 1:
    raise ValueError('Unsupported distribution version')
  rng = np.random.default_rng(seed)
  family = family or str(rng.choice(sorted(cfg['families'])))
  bounds = cfg['families'][family]
  profile = load_profile(bounds['profile'])
  profile.name = f"{cfg['name']}_{family}_{seed}"
  scale = float(rng.uniform(*bounds['scale']))
  v = profile.vehicle
  for name in ('lf', 'lr', 'track', 'length', 'width', 'front_overhang', 'rear_overhang', 'body_height'):
    setattr(v, name, getattr(v, name) * scale)
  v.mass *= scale ** 3
  v.iz *= scale ** 5
  v.cf *= scale ** 3
  v.cr *= scale ** 3
  v.drag *= scale ** 2
  for name in ('max_speed', 'tau_steer', 'tau_accel', 'max_accel', 'max_brake'):
    setattr(v, name, float(rng.uniform(*bounds[name])))
  v.max_steer = float(np.deg2rad(rng.uniform(*bounds['max_steer_deg'])))
  profile.mount = CameraMount(**{name: float(rng.uniform(*bounds[f'mount_{name}'])) for name in ('x', 'y', 'z')},
                             **{name: float(np.deg2rad(rng.uniform(*cfg[f'mount_{name}_deg'])))
                                for name in ('roll', 'pitch', 'yaw')})
  # Include a protruding camera support in the conservative collision footprint.
  v.front_overhang = max(v.front_overhang, profile.mount.x - v.lf + .03)
  v.rear_overhang = max(v.rear_overhang, -profile.mount.x - v.lr + .03)
  v.length = v.wheelbase + v.front_overhang + v.rear_overhang
  if profile.mount.z <= v.body_height:
    raise ValueError('Distribution places camera inside chassis; fix configuration bounds')
  profile.calibration_error = CameraMount(
    **{name: float(rng.uniform(-cfg['calibration_translation_m'], cfg['calibration_translation_m']))
       for name in ('x', 'y', 'z')},
    **{name: float(np.deg2rad(rng.uniform(-cfg['calibration_angle_deg'], cfg['calibration_angle_deg'])))
       for name in ('roll', 'pitch', 'yaw')})
  profile.motion = MountDynamics(cfg['motion_translation_m'], cfg['motion_angle_deg'],
                                float(rng.uniform(*cfg['motion_frequency_hz'])), rng.uniform(0, 2*np.pi, 6).tolist())
  width, height = cfg['sensor_resolutions'][int(rng.integers(len(cfg['sensor_resolutions'])))]
  hfov = float(np.deg2rad(rng.uniform(*cfg['sensor_hfov_deg'])))
  for camera in profile.cameras:
    camera.width, camera.height = width, height
    camera.fx = camera.fy = width / (2 * np.tan(hfov / 2))
    camera.cx, camera.cy = width / 2, height / 2
  provenance = {'version': cfg['version'], 'distribution': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'profile_sha256': hashlib.sha256(Path(bounds['profile']).read_bytes()).hexdigest(),
                'seed': seed, 'family': family}
  return profile, provenance
