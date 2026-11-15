"""Complete the guarded pilot, retaining the two completed probe episodes byte-for-byte."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path('data/sim/transfer_refresh_guarded_pilot_v7')


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
  source = Path('data/sim/transfer_refresh_pilot_v7')
  probe = Path('data/sim/transfer_refresh_guard_probe_v7')
  report = json.loads((probe/'report.json').read_text())
  assert report['gate'], 'Do not expand a failed guard probe'
  previous = json.loads((source/'manifest.json').read_text())
  specs = [{**s, 'intervention': {**s['intervention'], 'lateral_accel_disagreement_mps2': 2.}}
           for s in previous['specs']]
  retained_ids = {r['id'] for r in report['rows']}
  manifest = dict(schema=1, specs=specs, source_manifest_sha256=digest(source/'manifest.json'),
                  probe_manifest_sha256=digest(probe/'manifest.json'),
                  checkpoint_sha256=previous['checkpoint_sha256'],
                  retained_metadata_sha256={eid: digest(probe/eid/'meta.json') for eid in retained_ids})
  ROOT.mkdir(exist_ok=False)
  (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2))
  for eid in retained_ids:
    shutil.copytree(probe/eid, ROOT/eid)
    assert digest(ROOT/eid/'meta.json') == manifest['retained_metadata_sha256'][eid]
  (ROOT/'retained_rows.json').write_text(json.dumps(report['rows'], indent=2))
  with tarfile.open(ROOT/'source.tar.gz', 'w:gz') as archive:
    for folder in ('src', 'scripts', 'configs', 'tests', 'pyproject.toml', 'uv.lock'):
      archive.add(folder, filter=lambda item: None if '__pycache__' in item.name else item)
  return [s for s in specs if s['id'] not in retained_ids]
