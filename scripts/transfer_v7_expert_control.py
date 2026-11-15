"""Same-configuration expert controls for the failed collection quality gate."""
import hashlib
import json
from pathlib import Path

from fsop.sim.transfer_collection import collect


def run():
  source = Path('data/sim/transfer_refresh_pilot_v7')
  root = Path('data/sim/transfer_refresh_expert_control_v7')
  manifest = json.loads((source/'manifest.json').read_text())
  # No retuning or alternate tracks: one expert-only control for each exact pair.
  specs = [{**s, 'id': s['id'].replace('_current', '_expert'), 'collector': 'expert'}
           for s in manifest['specs'] if s['id'].endswith('_current')]
  root.mkdir(exist_ok=False)
  (root/'manifest.json').write_text(json.dumps(dict(specs=specs,
    paired_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()), indent=2))
  (root/'source.py').write_bytes(Path(__file__).read_bytes())
  rows = [collect(spec, str(root)) for spec in specs]
  (root/'index.json').write_text(json.dumps(dict(episodes=rows), indent=2))
  (root/'report.json').write_text(json.dumps(dict(rows=rows), indent=2))
  print(json.dumps(rows), flush=True)


if __name__ == '__main__':
  run()
