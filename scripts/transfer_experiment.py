"""Bounded, provenance-checked transfer experiment stages (executed on Modal)."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

BASELINE = 'runs/cuda_racing_v6/ckpt/best.pt'
DISTRIBUTION = 'configs/distributions/transfer_v1.yaml'


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_hashes():
  """A distribution's referenced profiles are part of its immutable geometry."""
  import yaml

  cfg = yaml.safe_load(Path(DISTRIBUTION).read_text())
  return {item['profile']: digest(item['profile']) for item in cfg['families'].values()}


def generate(name='transfer_pilot_v1', configurations=4, seconds=10.0):
  import cv2
  import numpy as np

  from fsop.data.sim_dataset import Episode
  from fsop.geometry.yuv import model_to_rgb
  from fsop.sim.batch_record import EpisodeSpec, _worker

  root = Path('data/sim') / name
  specs = []
  for split, offset, count in [('train', 0, configurations), ('val', 1000, max(2, configurations // 4))]:
    for i in range(count):
      family = 'rover' if i % 2 else 'formula'
      for recovery in (False, True):
        sid = offset + i
        specs.append(EpisodeSpec(f'{split}_{sid}_{int(recovery)}', 'acceleration:4.0',
                                 73191 + sid * 2 + int(recovery), 'light', False, seconds,
                                 noise=recovery, offset_max=.5 if family == 'rover' else .8,
                                 distribution=DISTRIBUTION, configuration_seed=73191 + sid,
                                 family=family, split=split))
  manifest = {'schema': 1, 'distribution_sha256': digest(DISTRIBUTION),
              'profile_sha256': profile_hashes(),
              'baseline_sha256': digest(BASELINE), 'specs': [asdict(s) for s in specs],
              'source_sha256': {str(p): digest(p) for p in sorted(Path('src/fsop').rglob('*.py'))}}
  root.mkdir(parents=True, exist_ok=True)
  manifest_path = root / 'manifest.json'
  if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
    raise ValueError('Dataset identity differs; use a distinct dataset name')
  manifest_path.write_text(json.dumps(manifest, indent=2))
  start = time.time()
  for spec in specs:
    if not (root / spec.episode_id / 'meta.json').exists():
      _worker(([spec], str(root)))
  rows, previews = [], []
  for spec in specs:
    ep = Episode(root / spec.episode_id)
    rows.append({'id': spec.episode_id, 'track_id': ep.meta['track_id'], 'discipline': ep.meta['discipline'],
                 'frames': ep.T, 'split': spec.split})
    if not spec.noise:
      previews.append(np.hstack([model_to_rgb(ep.frames[c][min(20, ep.T-1)]) for c in ('road', 'wide')]))
  (root / 'index.json').write_text(json.dumps({'episodes': rows}, indent=2))
  cv2.imwrite(str(root / 'preview.jpg'), np.vstack(previews)[..., ::-1])
  result = {'episodes': len(rows), 'frames': sum(r['frames'] for r in rows), 'seconds': time.time()-start,
            'manifest_sha256': digest(manifest_path)}
  (root / 'generation.json').write_text(json.dumps(result, indent=2))
  return result


def prepare_full(name='transfer_train_v1', configurations=16, seconds=20.0):
  """Fixed configuration partition; curved tracks plus explicit start/stop/recovery episodes."""
  import tarfile

  from fsop.sim.batch_record import EpisodeSpec
  from fsop.track.generate import generate_track
  from fsop.track.io import save_json

  root = Path('data/sim') / name
  root.mkdir(parents=True, exist_ok=True)
  track_root = Path('data/tracks') / name
  track_root.mkdir(parents=True, exist_ok=True)
  specs = []
  for split, offset, count in [('train', 0, configurations), ('val', 1000, max(4, configurations // 4))]:
    for i in range(count):
      sid = offset + i
      family = 'rover' if i % 2 else 'formula'
      path = track_root / f'{sid}.json'
      if not path.exists():
        track = generate_track(89000 + sid)
        track.laps = 1
        save_json(track, path)
      for case in ('drive', 'recovery', 'start', 'stop'):
        curved = case in ('drive', 'recovery')
        specs.append(EpisodeSpec(f'{split}_{sid}_{case}', str(path) if curved else 'acceleration:4.0',
                                 73191 + sid * 4 + ('drive', 'recovery', 'start', 'stop').index(case),
                                 'light', curved, seconds if curved else min(seconds, 12.0),
                                 noise=case == 'recovery', offset_max=.5 if family == 'rover' else .8,
                                 distribution=DISTRIBUTION, configuration_seed=83191 + sid,
                                 family=family, split=split, scenario='stop' if case == 'stop' else 'drive'))
  from fsop.data.sim_dataset import split_episodes
  replay = []
  pending_replay = []
  replay_root = Path('data/sim/racing_v6')
  for split, count in [('train', 4), ('val', 2)]:
    for eid in split_episodes(replay_root, split)[:count]:
      src = replay_root / eid
      dst = root / f'replay_{eid}'
      meta = json.loads((src / 'meta.json').read_text())
      meta['id'] = dst.name
      meta['frames_source'] = meta.get('frames_source', str(src / 'frames.zarr'))
      meta['split'] = split
      pending_replay.append((src, dst, meta))
      replay.append({'id': dst.name, 'source': str(src), 'split': split,
                     'meta_sha256': digest(src / 'meta.json'), 'plan_sha256': digest(src / 'plan.npy')})
  manifest = {'schema': 2, 'replay': replay, 'distribution_sha256': digest(DISTRIBUTION),
              'profile_sha256': profile_hashes(),
              'baseline_sha256': digest(BASELINE), 'specs': [asdict(s) for s in specs],
              'source_sha256': {str(p): digest(p) for p in sorted(Path('src/fsop').rglob('*.py'))}}
  mp = root / 'manifest.json'
  if mp.exists() and json.loads(mp.read_text()) != manifest:
    raise ValueError('Dataset identity differs; use a distinct dataset name')
  mp.write_text(json.dumps(manifest, indent=2))
  import shutil
  for src, dst, meta in pending_replay:
    if (dst / 'meta.json').exists():
      continue
    dst.mkdir(exist_ok=True)
    for file in src.iterdir():
      if file.is_file() and file.name != 'meta.json':
        shutil.copy2(file, dst / file.name)
    (dst / 'meta.json').write_text(json.dumps(meta, indent=2))
  (root / 'distribution.yaml').write_text(Path(DISTRIBUTION).read_text())
  if not (root / 'source.tar.gz').exists():
    with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
      for folder in ('src', 'configs', 'scripts'):
        archive.add(folder, filter=lambda info: None if '__pycache__' in info.name else info)
  return [asdict(s) for s in specs if not (root / s.episode_id / 'meta.json').exists()]


def finalize_full(name):
  root = Path('data/sim') / name
  manifest = json.loads((root / 'manifest.json').read_text())
  rows = []
  for spec in manifest['specs']:
    meta = json.loads((root / spec['episode_id'] / 'meta.json').read_text())
    if meta['frames'] < 2:
      raise ValueError(f"Empty episode: {meta['id']}")
    rows.append({'id': meta['id'], 'track_id': meta['track_id'], 'discipline': meta['discipline'],
                 'frames': meta['frames'], 'split': spec['split'], 'scenario': spec['scenario']})
  for item in manifest.get('replay', []):
    meta = json.loads((root / item['id'] / 'meta.json').read_text())
    rows.append({'id': item['id'], 'track_id': meta['track_id'], 'discipline': meta['discipline'],
                 'frames': meta['frames'], 'split': item['split'], 'scenario': 'legacy_replay'})
  (root / 'index.json').write_text(json.dumps({'episodes': rows}, indent=2))
  return {'episodes': len(rows), 'frames': sum(r['frames'] for r in rows)}


def prepare_test(name='transfer_test_v1'):
  """Call only after validation selection; final configurations and seeds are immutable."""
  import yaml

  from fsop.eval.transfer import transfer_profile, transfer_track
  from fsop.sim.batch_record import EpisodeSpec
  from fsop.track.io import save_json

  cfg = yaml.safe_load(Path('configs/eval/transfer_v1.yaml').read_text())
  root = Path('data/sim') / name
  root.mkdir(parents=True, exist_ok=True)
  assets = Path('data/tracks') / name
  assets.mkdir(parents=True, exist_ok=True)
  specs, configurations, profiles = [], {}, []
  for item in cfg['configurations']:
    profile = transfer_profile(item)
    pp = assets / f"{item['name']}.yaml"
    profiles.append((pp, yaml.safe_dump(profile.to_dict())))
    for j, (track_seed, scene_seed) in enumerate(zip(cfg['track_seeds'], cfg['scene_seeds'], strict=True)):
      tp = assets / f'{track_seed}.json'
      if not tp.exists():
        save_json(transfer_track(track_seed), tp)
      eid = f"{item['name']}_{j}"
      specs.append(EpisodeSpec(eid, str(tp), scene_seed, 'off', True, 12., noise=False,
                               profile=str(pp), physical_camera=True, split='test', expert_mode='configured'))
      configurations[eid] = item
  manifest = {'schema': 1, 'eval_sha256': digest('configs/eval/transfer_v1.yaml'),
              'profile_sha256': profile_hashes(),
              'specs': [asdict(s) for s in specs], 'configurations': configurations,
              'source_sha256': {str(p): digest(p) for p in sorted(Path('src/fsop').rglob('*.py'))}}
  mp = root / 'manifest.json'
  if mp.exists() and json.loads(mp.read_text()) != manifest:
    raise ValueError('Final dataset identity changed; do not reuse its name')
  mp.write_text(json.dumps(manifest, indent=2))
  for path, contents in profiles:
    path.write_text(contents)
  return [asdict(s) for s in specs if not (root / s.episode_id / 'meta.json').exists()]


def finalize_test(name='transfer_test_v1'):
  root = Path('data/sim') / name
  manifest = json.loads((root / 'manifest.json').read_text())
  rows = []
  for spec in manifest['specs']:
    meta = json.loads((root / spec['episode_id'] / 'meta.json').read_text())
    config = manifest['configurations'][spec['episode_id']]
    rows.append({'id': meta['id'], 'track_id': meta['track_id'], 'discipline': meta['discipline'],
                 'frames': meta['frames'], 'split': 'test',
                 'configuration': config['name'], 'regime': config['regime']})
  (root / 'index.json').write_text(json.dumps({'episodes': rows}, indent=2))
  return {'episodes': len(rows), 'frames': sum(r['frames'] for r in rows)}


def audit_dataset(dataset):
  import numpy as np

  root = Path(dataset)
  rows = []
  for item in json.loads((root / 'index.json').read_text())['episodes']:
    path = root / item['id']
    meta = json.loads((path / 'meta.json').read_text())
    plans = np.load(path / 'plan.npy')
    states = np.load(path / 'states.npy')
    if not np.isfinite(plans).all() or not np.isfinite(states).all():
      raise ValueError(f'Nonfinite supervision: {path}')
    if plans.shape != (meta['frames'], 33, 15):
      raise ValueError(f'Invalid plan shape: {path}')
    safety = meta.get('label_planner', {}).get('rollout_safety', {})
    rows.append({'episode': item['id'], 'split': item['split'], 'frames': meta['frames'],
                 'unresolved_label_frames': safety.get('unresolved_frames', 0),
                 'generation_timing': meta.get('generation_timing'),
                 'contacts': meta['summary']['doo'], 'peak_speed': float(states[:, 4].max())})
  total = sum(r['frames'] for r in rows)
  unresolved = sum(r['unresolved_label_frames'] for r in rows)
  report = {'frames': total, 'unresolved_label_frames': unresolved,
            'unresolved_fraction': unresolved / max(total, 1), 'episodes': rows}
  (root / 'quality.json').write_text(json.dumps(report, indent=2))
  if report['unresolved_fraction'] > .1:
    raise ValueError(f'Label safety gate failed: {unresolved}/{total} frames unresolved')
  return report


def preview_dataset(dataset):
  """Recorded observations with physically projected COG-relative expert plans."""
  import cv2
  import numpy as np

  from fsop.data.sim_dataset import Episode
  from fsop.geometry.camera import CAMERA_MODELS, project_world
  from fsop.geometry.yuv import model_to_rgb

  root = Path(dataset)
  panels = []
  seen = set()
  for row in json.loads((root / 'index.json').read_text())['episodes']:
    ep = Episode(root / row['id'])
    seed = ep.meta.get('distribution', {}).get('seed')
    if seed is None or seed in seen:
      continue
    seen.add(seed)
    k = min(25, ep.T - 1)
    tiles = []
    for camera, model in CAMERA_MODELS.items():
      image = model_to_rgb(ep.frames[camera][k]).copy()
      points = ep.plan[k, :, :3]
      uv, _, valid = project_world(points, np.eye(4), ep.camera_pose(k), model)
      valid &= ep.observable(points, k, camera)
      for point in uv[valid]:
        cv2.circle(image, tuple(point.astype(int)), 3, (255, 60, 80), -1)
      tiles.append(image)
    panel = np.hstack(tiles)
    mount = ep.meta['mount']
    label = f"{row['id']} xyz=({mount['x']:.2f},{mount['y']:.2f},{mount['z']:.2f})"
    cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    panels.append(cv2.resize(panel, (768, 192)))
  output = root / 'observations.jpg'
  cv2.imwrite(str(output), np.vstack(panels)[..., ::-1])
  return str(output)


def derive_minimal(source='runs/transfer_anchored_v1/ckpt/best.pt', name='transfer_minimal_v1'):
  """Validation-chosen specialization: remove angle conditioning, without another training run."""
  import tarfile

  import numpy as np
  import torch
  import yaml

  from fsop.data.sim_dataset import Episode
  from fsop.eval.open_loop import run_episode_model
  from fsop.geometry.camera import CameraMount
  from fsop.model.build import load_fs
  from fsop.model.conditioning import context_fields, geometry_context, prune_orientation_state
  from fsop.train.device import pick_device

  root = Path('runs') / name
  if root.exists():
    raise ValueError('Derived run already exists; do not overwrite provenance')
  saved = torch.load(source, map_location='cpu', weights_only=False)
  if saved['input_contract']['geometry_version'] != 1:
    raise ValueError('Expected the v1 anchored checkpoint')
  saved['model'] = prune_orientation_state(saved['model'])
  saved['input_contract']['geometry_version'] = 2
  saved['input_contract']['fields'] = list(context_fields(2))
  saved['cfg'] = {**saved['cfg'], 'name': name, 'context_version': 2}
  for key in ('optimizer', 'torch_rng', 'numpy_rng', 'cuda_rng', 'best'):
    saved.pop(key, None)
  saved['inference_only'] = True
  saved['derived_from'] = {'checkpoint': source, 'sha256': digest(source), 'step': saved['step'],
                           'method': 'specialize camera angles to zero; remove their first-layer columns'}
  (root / 'ckpt').mkdir(parents=True)
  destination = root / 'ckpt/best.pt'
  torch.save(saved, destination)
  (root / 'config.yaml').write_text(yaml.safe_dump(saved['cfg']))
  provenance = {**saved['derived_from'], 'source_sha256':
                {str(p): digest(p) for p in sorted(Path('src/fsop').rglob('*.py'))}}
  (root / 'provenance.json').write_text(json.dumps(provenance, indent=2))
  with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
    for folder in ('src', 'configs', 'scripts', 'tests'):
      archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)
  # The same recorded validation sequence checks specialization across time, not only a synthetic tensor.
  device = pick_device()
  ep = Episode('data/sim/transfer_train_v1/val_1001_drive')
  original, pruned = load_fs(source).to(device).eval(), load_fs(destination).to(device).eval()
  context = geometry_context(CameraMount(**ep.meta['calibrated_mount']), ep.vehicle)
  context[3:6] = 0
  expected = run_episode_model(original, ep, device, bev=False, geometry_override=context)
  actual = run_episode_model(pruned, ep, device, bev=False)
  difference = float(np.abs(expected['mu'] - actual['mu']).max())
  result = {'checkpoint': str(destination), 'source_checkpoint': source,
            'specialization_max_abs': difference, 'validation_episode': str(ep.path), 'frames': ep.T}
  (root / 'specialization.json').write_text(json.dumps(result, indent=2))
  if difference > .001:
    raise ValueError(f'Specialization parity failed: {result}')
  print(result)
  return result
