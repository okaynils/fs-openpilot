"""One-variable safeguard probe on the failed training configuration pair."""
import hashlib
import json
import tarfile
from pathlib import Path

from fsop.sim.transfer_collection import collect


def run():
  source = Path('data/sim/transfer_refresh_pilot_v7')
  root = Path('data/sim/transfer_refresh_guard_probe_v7')
  manifest = json.loads((source/'manifest.json').read_text())
  specs = [{**s, 'intervention': {**s['intervention'], 'lateral_accel_disagreement_mps2': 2.}}
           for s in manifest['specs'] if s['configuration'] == 'train_v2_00']
  assert len(specs) == 2
  root.mkdir(exist_ok=False)
  (root/'manifest.json').write_text(json.dumps(dict(specs=specs,
    paired_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
    checkpoint_sha256=manifest['checkpoint_sha256']), indent=2))
  with tarfile.open(root/'source.tar.gz', 'w:gz') as archive:
    for folder in ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'uv.lock'):
      archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)
  rows = [collect(spec, str(root)) for spec in specs]
  (root/'report.json').write_text(json.dumps(dict(rows=rows,
    gate=all(r['contacts'] <= 2 and r['valid_frames']/r['frames'] >= .95 for r in rows)), indent=2))
  print(json.dumps(rows), flush=True)


if __name__ == '__main__':
  run()
