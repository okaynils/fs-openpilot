"""Replace only v2 on-policy episodes; preserve exact replay/expert metadata and caches."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import yaml

SOURCE = Path('data/sim/transfer_train_v2')
NAME = 'transfer_reference_train_v3'
CACHE = 'transfer_baseline_strict_v2_full'


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(name=NAME, policy_path='configs/distributions/transfer_collection_reference_v3.yaml'):
  root = Path('data/sim') / name
  previous = json.loads((SOURCE / 'manifest.json').read_text())
  rows = json.loads((SOURCE / 'index.json').read_text())['episodes']
  policy = yaml.safe_load(Path(policy_path).read_text())
  specs = [{**spec, 'intervention': policy['intervention'], 'collection_policy': policy}
           for spec in previous['specs'] if spec['collector'] == 'dagger']
  retained = [row for row in rows if row.get('collector') != 'dagger']
  assert len(specs) == 24 and len(retained) == 114
  source = {str(p): digest(p) for folder in ('src', 'configs', 'scripts', 'tests')
            for p in sorted(Path(folder).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
  identity = dict(schema=1, source_dataset=str(SOURCE), source_manifest_sha256=digest(SOURCE/'manifest.json'),
                  specs=specs, retained_metadata_sha256={r['id']:digest(SOURCE/r['id']/'meta.json') for r in retained},
                  source_sha256=source, collector_checkpoint_sha256=digest(specs[0]['checkpoint']),
                  cache_reuse='Unchanged episode contents/metadata only; build_cache validates episode/vision hashes')
  root.mkdir(parents=True, exist_ok=True)
  manifest = root/'manifest.json'
  if manifest.exists() and json.loads(manifest.read_text()) != identity:
    raise ValueError('Dataset identity changed; use a distinct name')
  manifest.write_text(json.dumps(identity, indent=2))
  for row in retained:
    src, dst = SOURCE/row['id'], root/row['id']
    # Copy unchanged bytes, including external frames_source references. Do not rebrand
    # old episodes or silently reuse a feature cache with altered geometry metadata.
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=lambda directory,names: {n for n in names
                                                   if n.startswith('cache_') and n != f'cache_{CACHE}'})
    assert digest(dst/'meta.json') == identity['retained_metadata_sha256'][row['id']]
  (root/'retained_rows.json').write_text(json.dumps(retained,indent=2))
  if not (root/'source.tar.gz').exists():
    with tarfile.open(root/'source.tar.gz','w:gz') as archive:
      for folder in ('src','configs','scripts','tests','pyproject.toml','uv.lock'):
        archive.add(folder,filter=lambda item:None if '__pycache__' in item.name else item)
  return specs


def finish(rows, name=NAME):
  root = Path('data/sim')/name
  retained = json.loads((root/'retained_rows.json').read_text())
  if {r['id'] for r in rows} != {s['id'] for s in json.loads((root/'manifest.json').read_text())['specs']}:
    raise ValueError('Incomplete replacement collection')
  by_id = {r['id']:r for r in retained+rows}
  original = json.loads((SOURCE/'index.json').read_text())['episodes']
  assert set(by_id) == {r['id'] for r in original}
  (root/'index.json').write_text(json.dumps({'episodes':[by_id[r['id']] for r in original]},indent=2))
  report = dict(episodes=len(retained)+len(rows), replaced_episodes=len(rows),
                frames=sum(r['frames'] for r in rows), valid_frames=sum(r['valid_frames'] for r in rows),
                worker_seconds=sum(r['seconds'] for r in rows),rows=rows,manifest_sha256=digest(root/'manifest.json'))
  (root/'generation.json').write_text(json.dumps(report,indent=2))
  return report
