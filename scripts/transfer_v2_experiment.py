"""Versioned design and immutable dataset assembly for the recovery-transfer experiment."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import yaml

from fsop.geometry.camera import CameraMount
from fsop.geometry.mounting import MountDynamics
from fsop.profile import load_profile
from fsop.sim.distribution import sample_profile

DISTRIBUTION = Path('configs/distributions/transfer_v2.yaml')
PATTERNS = [(0, 0, .15), (1, 1, 0), (.75, 0, 1), (0, 1, .85),
            (.5, .5, .5), (1, .5, .65), (.3, 1, 0), (.8, .25, .25),
            (.18, .7, .2), (.82, .3, .8), (.35, .35, .35), (.65, .65, .65)]


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(seed, family, fractions, name, regime='interpolation', moving=False):
  distribution = yaml.safe_load(DISTRIBUTION.read_text())
  bounds = distribution['families'][family]
  profile, provenance = sample_profile(DISTRIBUTION, seed, family)
  base = load_profile(bounds['profile'])
  for axis, fraction in zip(('x', 'y', 'z'), fractions, strict=True):
    low, high = bounds[f'mount_{axis}']
    setattr(profile.mount, axis, float(low + fraction * (high-low)))
  v = profile.vehicle
  scale = v.wheelbase / base.vehicle.wheelbase
  v.front_overhang = max(base.vehicle.front_overhang * scale, profile.mount.x - v.lf + .03)
  v.rear_overhang = max(base.vehicle.rear_overhang * scale, -profile.mount.x - v.lr + .03)
  v.length = v.wheelbase + v.front_overhang + v.rear_overhang
  for camera in profile.cameras:
    ratio = 640 / camera.width
    camera.width, camera.height = 640, 360
    camera.fx = float(camera.fx * ratio)
    camera.fy = float(camera.fy * ratio)
    camera.cx, camera.cy = 320., 180.
  if not moving:
    profile.motion = MountDynamics()
    profile.calibration_error = CameraMount(z=0.)
  profile.name = name
  return {'name': name, 'family': family, 'regime': regime, 'profile': profile.to_dict(),
          'design': {'seed': seed, 'placement_fractions': list(fractions), 'distribution': provenance}}


def write_design():
  """Freeze metadata only. Final test observations are not generated or evaluated here."""
  train = [configuration(184191+i, 'formula' if i % 2 == 0 else 'rover', PATTERNS[i//2],
                         f'train_v2_{i:02d}', moving=True) for i in range(24)]
  val = [configuration(284191+i, 'formula' if i < 2 else 'rover', fractions, f'val_v2_{i}')
         for i, fractions in enumerate([(.13, .23, .28), (.87, .75, .82),
                                         (.86, .26, .18), (.19, .81, .82)])]
  final = [configuration(384191+i, family, fractions, f'test_v2_{i}', regime)
           for i, (family, fractions, regime) in enumerate([
             ('formula', (.82, .2, .21), 'interpolation'),
             ('formula', (.12, .82, .87), 'interpolation'),
             ('rover', (.91, .85, .14), 'interpolation'),
             ('rover', (.09, .15, .77), 'interpolation'),
             ('formula', (-.10, 1.1, 1.15), 'extrapolation'),
             ('rover', (1.1, -.1, -.04), 'extrapolation')])]
  for path, data in [
    ('configs/distributions/transfer_v2_design.yaml', {'version': 1, 'configurations': train}),
    ('configs/eval/transfer_v2_val.yaml', {'version': 2, 'track_seeds': [98011],
                                       'scene_seeds': [5101], 'configurations': val}),
    ('configs/eval/transfer_v2_test.yaml', {'version': 2, 'track_seeds': [99111, 99112],
                                        'scene_seeds': [6101, 6102], 'configurations': final})]:
    p = Path(path)
    text = yaml.safe_dump(data, sort_keys=False)
    if p.exists() and p.read_text() != text:
      raise ValueError(f'Design already exists with different contents: {p}')
    p.write_text(text)


def prepare(name, pilot=False):
  root = Path('data/sim') / name
  train = yaml.safe_load(Path('configs/distributions/transfer_v2_design.yaml').read_text())['configurations']
  if pilot:
    train = [train[i] for i in (0, 3, 12, 15)]
  validation = yaml.safe_load(Path('configs/eval/transfer_v2_val.yaml').read_text())['configurations']
  specs = []
  for split, configs in [('train', train), ('val', validation)]:
    for i, config in enumerate(configs):
      for collector in (('expert', 'dagger') if split == 'train' else ('expert',)):
        specs.append({'id': f'{config["name"]}_{collector}', 'configuration': config['name'],
                      'profile': config['profile'], 'split': split, 'collector': collector,
                      'checkpoint': 'runs/transfer_minimal_v1/ckpt/best.pt',
                      'track_seed': 880001 + i % 4 if split == 'train' else 98011,
                      'scene_seed': 7101 + i if split == 'train' else 5101,
                      'seconds': 12. if pilot else 24., 'random_start': True})
  replay_root = Path('data/sim/transfer_train_v1')
  replay = json.loads((replay_root / 'index.json').read_text())['episodes']
  if pilot:
    replay = [r for r in replay if r['id'].startswith('replay_')]
  source = {str(p): digest(p) for folder in ('src', 'configs', 'scripts')
            for p in sorted(Path(folder).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
  identity = {'schema': 1, 'pilot': pilot, 'specs': specs, 'source_sha256': source,
              'replay_manifest_sha256': digest(replay_root / 'manifest.json'),
              'collector_checkpoint_sha256': digest('runs/transfer_minimal_v1/ckpt/best.pt')}
  root.mkdir(parents=True, exist_ok=True)
  manifest = root / 'manifest.json'
  if manifest.exists() and json.loads(manifest.read_text()) != identity:
    raise ValueError('Dataset identity changed; choose a distinct name')
  manifest.write_text(json.dumps(identity, indent=2))
  replay_rows = []
  for row in replay:
    src = replay_root / row['id']
    dst = root / f'replay_v1_{row["id"]}'
    if not (dst / 'meta.json').exists():
      dst.mkdir(exist_ok=True)
      for file in src.iterdir():
        if file.is_file() and file.name != 'meta.json':
          shutil.copy2(file, dst / file.name)
      meta = json.loads((src / 'meta.json').read_text())
      meta.update(id=dst.name, frames_source=str(src / 'frames.zarr') if (src/'frames.zarr').exists()
                  else meta['frames_source'], collector='replay')
      (dst / 'meta.json').write_text(json.dumps(meta, indent=2))
    replay_rows.append({**row, 'id': dst.name, 'collector': 'replay'})
  (root / 'replay_rows.json').write_text(json.dumps(replay_rows, indent=2))
  if not (root / 'source.tar.gz').exists():
    with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
      for folder in ('src', 'configs', 'scripts', 'tests'):
        archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)
  return specs


def finish(name, rows):
  root = Path('data/sim') / name
  replay = json.loads((root / 'replay_rows.json').read_text())
  (root / 'index.json').write_text(json.dumps({'episodes': replay + rows}, indent=2))
  result = {'episodes': len(replay)+len(rows), 'new_frames': sum(r['frames'] for r in rows),
            'new_valid_frames': sum(r['valid_frames'] for r in rows),
            'worker_seconds': sum(r['seconds'] for r in rows),
            'rows': rows, 'manifest_sha256': digest(root / 'manifest.json')}
  (root / 'generation.json').write_text(json.dumps(result, indent=2))
  return result


if __name__ == '__main__':
  write_design()
