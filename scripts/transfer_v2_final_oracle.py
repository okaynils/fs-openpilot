"""Post-selection oracle diagnosis; never used for training or checkpoint selection."""
import json
import runpy
from pathlib import Path

import yaml


def diagnose():
  if not Path('benchmarks/transfer_v2/selection.json').exists():
    raise ValueError('Selection must be frozen before opening final cases')
  api = runpy.run_path('scripts/transfer_diagnostics.py')
  suite = yaml.safe_load(Path('configs/eval/transfer_v2_test.yaml').read_text())
  root = Path('benchmarks/transfer_v2/oracle_final')
  root.mkdir(exist_ok=True)
  rows = []
  for configuration in suite['configurations']:
    for i, (track, scene) in enumerate(zip(suite['track_seeds'], suite['scene_seeds'], strict=True)):
      path = root / f"{configuration['name']}_{track}_{scene}.json"
      if path.exists():
        raise ValueError(f'Preserve existing oracle run: {path}')
      results = api['oracle_tracking'](configuration, path, track, scene, bool(i),
                                       180. if configuration['family'] == 'rover' else 100.)
      for result in results:
        row = dict(configuration=configuration['name'], track_seed=track, scene_seed=scene,
                   recovery=bool(i), mode=result['mode'], **result['summary'])
        rows.append(row)
        print(row, flush=True)
  (root / 'summary.json').write_text(json.dumps({'execution': 'local CPU, no rendering', 'results': rows}, indent=2))


if __name__ == '__main__':
  diagnose()
