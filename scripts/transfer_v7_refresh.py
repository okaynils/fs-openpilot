"""Paired collection pilot on training configurations, preserving checkpoint provenance."""
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('data/sim/transfer_refresh_pilot_v7')


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
  settings = yaml.safe_load(Path('configs/distributions/transfer_refresh_v7.yaml').read_text())
  design = yaml.safe_load(Path(settings['source_design']).read_text())['configurations']
  specs = []
  for index in settings['configuration_indices']:
    config = design[index]
    for arm, checkpoint in settings['collectors'].items():
      specs.append(dict(id=f"{config['name']}_{arm}", configuration=config['name'],
                        profile=config['profile'], split='train', collector='dagger', checkpoint=checkpoint,
                        track_seed=settings['track_seed_base']+index, scene_seed=settings['scene_seed_base']+index,
                        seconds=settings['seconds'], random_start=settings['random_start'],
                        intervention=settings['intervention']))
  manifest = dict(schema=1, settings=settings, specs=specs,
                  checkpoint_sha256={p: digest(p) for p in settings['collectors'].values()},
                  source_sha256={str(p): digest(p) for folder in ('src', 'scripts', 'configs')
                                 for p in sorted(Path(folder).rglob('*'))
                                 if p.is_file() and '__pycache__' not in str(p)})
  ROOT.mkdir(parents=True, exist_ok=True)
  path = ROOT/'manifest.json'
  if path.exists() and json.loads(path.read_text()) != manifest:
    raise ValueError('Preserve previous collection identity; choose a new dataset name')
  path.write_text(json.dumps(manifest, indent=2))
  if not (ROOT/'source.tar.gz').exists():
    with tarfile.open(ROOT/'source.tar.gz', 'w:gz') as archive:
      for folder in ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'uv.lock'):
        archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)
  return specs


def finish(rows, root=ROOT):
  manifest = json.loads((root/'manifest.json').read_text())
  assert {r['id'] for r in rows} == {s['id'] for s in manifest['specs']}
  summaries = []
  for row in rows:
    path = root/row['id']
    states = np.load(path/'states.npy')
    controls = np.load(path/'controls.npy')
    takeover = np.load(path/'expert_intervention.npy')
    valid = np.load(path/'label_valid.npy')
    summaries.append(dict(**row, initial_speed=float(states[0, 4]),
                          low_speed_frames=int((states[:, 4] <= .5).sum()),
                          intervention_transitions=int(np.count_nonzero(np.diff(takeover.astype(int)))),
                          valid_fraction=float(valid.mean()),
                          applied_vs_expert_steering_mae=float(np.abs(controls[:, 0]-controls[:, 2]).mean())))
  (root/'index.json').write_text(json.dumps(dict(episodes=rows), indent=2))
  report = dict(rows=summaries, manifest_sha256=digest(root/'manifest.json'),
                worker_seconds=sum(r['seconds'] for r in rows),
                limitation='Training configurations only; collection safety interventions are privileged. '
                           'Applied steering equals expert during takeover; not a raw policy error metric.')
  (root/'report.json').write_text(json.dumps(report, indent=2))
  return report
