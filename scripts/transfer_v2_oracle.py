"""Validation-only oracle separates trajectory tracking from learned visual prediction."""
import json
import runpy
from pathlib import Path

import yaml

api = runpy.run_path('scripts/transfer_diagnostics.py')
suite = yaml.safe_load(Path('configs/eval/transfer_v2_val.yaml').read_text())
root = Path('benchmarks/transfer_v2/oracle_validation')
root.mkdir(exist_ok=True)
rows = []
for i, configuration in enumerate(suite['configurations']):
  path = root / f"{configuration['name']}.json"
  if path.exists():
    raise ValueError(f'Preserve existing oracle run: {path}')
  results = api['oracle_tracking'](configuration, path, suite['track_seeds'][0], suite['scene_seeds'][0],
                                   bool(i % 2), 180. if configuration['family'] == 'rover' else 100.)
  for result in results:
    row = dict(configuration=configuration['name'], mode=result['mode'], **result['summary'])
    rows.append(row)
    print(row, flush=True)
(root / 'summary.json').write_text(json.dumps({'execution': 'local CPU, no rendering', 'results': rows}, indent=2))
