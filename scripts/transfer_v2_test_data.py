"""Prepare independent test recordings only after a driving-validation selection exists."""
import hashlib
import json
import tarfile
from pathlib import Path

import yaml


def prepare(selection, name):
  selection = Path(selection)
  selected = json.loads(selection.read_text())
  checkpoint = Path(selected['selected_candidate'])
  if not checkpoint.exists():
    raise ValueError('Selected validation checkpoint must exist before opening final tests')
  suite = yaml.safe_load(Path('configs/eval/transfer_v2_test.yaml').read_text())
  specs = []
  for configuration in suite['configurations']:
    for i, (track, scene) in enumerate(zip(suite['track_seeds'], suite['scene_seeds'], strict=True)):
      specs.append({'id': f"{configuration['name']}_{i}", 'configuration': configuration['name'],
                    'regime': configuration['regime'], 'profile': configuration['profile'],
                    'split': 'test', 'collector': 'expert', 'checkpoint': str(checkpoint),
                    'track_seed': track, 'scene_seed': scene, 'seconds': 24., 'random_start': True})
  root = Path('data/sim') / name
  root.mkdir(parents=True, exist_ok=True)
  identity = {'suite': suite, 'selection': selected,
              'selected_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(), 'specs': specs,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ('src', 'configs', 'scripts')
                                for p in sorted(Path(folder).rglob('*'))
                                if p.is_file() and '__pycache__' not in str(p)}}
  path = root / 'manifest.json'
  if path.exists() and json.loads(path.read_text()) != identity:
    raise ValueError('Final-test identity changed; retain the original test dataset')
  path.write_text(json.dumps(identity, indent=2))
  if not (root / 'source.tar.gz').exists():
    with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
      for folder in ('src', 'configs', 'scripts', 'tests'):
        archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)
  return specs


def finish(name, specs, rows):
  root = Path('data/sim') / name
  regimes = {s['id']: s['regime'] for s in specs}
  rows = [{**r, 'regime': regimes[r['id']]} for r in rows]
  (root / 'index.json').write_text(json.dumps({'episodes': rows}, indent=2))
  report = {'rows': rows, 'worker_seconds': sum(r['seconds'] for r in rows),
            'manifest_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
  (root / 'generation.json').write_text(json.dumps(report, indent=2))
  return {'episodes': len(rows), 'frames': sum(r['frames'] for r in rows)}
