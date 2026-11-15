"""Reuse paired driving references only when all inference/simulation code is unchanged."""
import hashlib
import json
import sys
from pathlib import Path

# This module is called only by collection jobs, never the evaluator/runtime.
COLLECTION_ONLY = {'src/fsop/sim/transfer_collection.py'}


def merge(reference, reference_cases, candidate, candidate_cases, output):
  paths = [Path(reference), Path(candidate)]
  cases = [Path(reference_cases), Path(candidate_cases)]
  first, second = [json.loads(p.read_text()) for p in paths]
  for key in ('suite', 'recovery_rule', 'time_limits_seconds'):
    if first[key] != second[key]:
      raise ValueError(f'Unpaired evaluation contract: {key}')
  source_differences = []
  for config in first['suite']['configurations']:
    for track, scene in zip(first['suite']['track_seeds'], first['suite']['scene_seeds'], strict=True):
      name = f"{config['name']}_{track}_{scene}.json"
      identities = [json.loads((root/name).read_text())['identity'] for root in cases]
      for key in ('configuration', 'track_seed', 'scene_seed', 'recovery', 'max_seconds'):
        if identities[0][key] != identities[1][key]:
          raise ValueError(f'Unpaired {name}: {key}')
      hashes = [{k:v for k,v in i['source_sha256'].items() if k.startswith('src/fsop/')} for i in identities]
      changed = {k for k in set(hashes[0]) | set(hashes[1]) if hashes[0].get(k) != hashes[1].get(k)}
      if changed - COLLECTION_ONLY:
        raise ValueError(f'Inference/simulation source changed: {sorted(changed)}')
      source_differences.append(dict(case=name, collection_only_differences=sorted(changed)))
  unique = {}
  for row in first['results']+second['results']:
    key = tuple(row[k] for k in ('checkpoint','configuration','track_seed','scene_seed','recovery'))
    if key in unique:
      if row['checkpoint'] != 'expert':
        raise ValueError('Duplicate learned-policy evaluation')
      for field in ('pass','complete','doo','oc','n_laps','dq'):
        if row[field] != unique[key][field]:
          raise ValueError(f'Expert repeat disagrees: {key} {field}')
    else:
      unique[key] = row
  report = {**first,'results':list(unique.values()),
            'checkpoints':list(dict.fromkeys(first['checkpoints']+second['checkpoints'])),
            'reference_inputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            'source_audit':source_differences,
            'reference_reuse':'Identical complete configurations/seeds/limits and inference/simulation source; '
                              'only the unused collection module may differ. Repeated expert outcomes checked.'}
  output=Path(output)
  if output.exists():
    raise ValueError('Preserve existing merged report')
  output.write_text(json.dumps(report,indent=2))
  return report


if __name__ == '__main__':
  merge(*sys.argv[1:])
