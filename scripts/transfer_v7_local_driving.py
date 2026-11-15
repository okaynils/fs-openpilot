"""Supplemental CPU driving comparison; does not replace the interrupted paired Modal study."""
import hashlib
import json
import tarfile
from pathlib import Path

import torch
import yaml

from fsop.eval.transfer import closed_loop_case


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
  torch.set_num_threads(2)
  root = Path('benchmarks/transfer_v7/local_driving')
  root.mkdir(exist_ok=False)
  checkpoints = ['runs/cuda_racing_v6/ckpt/best.pt', 'runs/transfer_minimal_v1/ckpt/best.pt',
                 'artifacts/experiments/transfer_refresh_current_pilot_v7/step_000300.pt',
                 'artifacts/export/transfer_refresh_current_pilot_v7/candidate.onnx']
  suite_path = Path('configs/eval/transfer_v2_val.yaml')
  suite = yaml.safe_load(suite_path.read_text())
  identity = dict(suite=suite, suite_sha256=digest(suite_path),
                  checkpoints={p: digest(p) for p in checkpoints}, backend='CPU',
                  source_sha256={str(p): digest(p) for p in sorted(Path('src/fsop').rglob('*.py'))})
  (root/'identity.json').write_text(json.dumps(identity, indent=2))
  with tarfile.open(root/'source.tar.gz', 'w:gz') as archive:
    for folder in ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'uv.lock'):
      archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)
  results = []
  # Preserve complete case results after each configuration. No final holdout is read.
  for i, configuration in enumerate(suite['configurations']):
    rows = closed_loop_case(configuration, checkpoints, 98011, 5101, bool(i%2),
                            180. if configuration['family'] == 'rover' else 100., device='cpu')
    results.extend(rows)
    (root/f"{configuration['name']}.json").write_text(json.dumps(rows, indent=2))
  (root/'report.json').write_text(json.dumps(dict(identity=identity, results=results,
    limitation='Supplemental local CPU comparison, not the required paired Modal GPU evaluation. '
               'The interrupted previous-refresh arm is absent. Renderer and hardware differ from prior Modal '
               'runs, so only within-this-run comparisons are controlled. Mission assistance is reported per row.'),
                                            indent=2))


if __name__ == '__main__':
  run()
