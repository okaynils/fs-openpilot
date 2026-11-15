"""Combine the bounded follow-up with paired reference results without rerunning references."""
import hashlib
import json
import sys
from pathlib import Path


def merge(first_path, second_path, output):
  first_path, second_path, output = Path(first_path), Path(second_path), Path(output)
  first, second = [json.loads(p.read_text()) for p in (first_path, second_path)]
  for key in ('suite', 'recovery_rule', 'time_limits_seconds'):
    if first[key] != second[key]:
      raise ValueError(f'Unpaired evaluation contract: {key}')
  for configuration in first['suite']['configurations']:
    for track, scene in zip(first['suite']['track_seeds'], first['suite']['scene_seeds'], strict=True):
      case = f"{configuration['name']}_{track}_{scene}.json"
      identities = [json.loads((p.parent / case).read_text())['identity'] for p in (first_path, second_path)]
      for key in ('configuration', 'track_seed', 'scene_seed', 'recovery', 'max_seconds'):
        if identities[0][key] != identities[1][key]:
          raise ValueError(f'Unpaired case {case}: {key}')
      for path, digest in identities[0]['source_sha256'].items():
        if path.startswith('src/fsop/') and identities[1]['source_sha256'].get(path) != digest:
          raise ValueError(f'Evaluator/model/simulator code changed: {path}')
  unique = {}
  for row in first['results'] + second['results']:
    key = tuple(row[k] for k in ('checkpoint', 'configuration', 'track_seed', 'scene_seed', 'recovery'))
    if key in unique:
      if row['checkpoint'] != 'expert':
        raise ValueError('Duplicate learned-policy evaluation')
      for field in ('pass', 'complete', 'doo', 'oc', 'n_laps', 'dq'):
        if unique[key][field] != row[field]:
          raise ValueError(f'Expert repeat disagrees: {key} {field}')
    else:
      unique[key] = row
  report = {**first, 'results': list(unique.values()),
            'checkpoints': list(dict.fromkeys(first['checkpoints'] + second['checkpoints'])),
            'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in (first_path, second_path)],
            'reference_reuse': 'same complete profiles, seeds, recovery, limits and src/fsop source hashes; '
                               'duplicate expert outcomes checked, first record retained'}
  output.parent.mkdir(parents=True, exist_ok=True)
  if output.exists():
    raise ValueError('Preserve existing merged report')
  output.write_text(json.dumps(report, indent=2))
  print('merged', len(report['results']), 'unique evaluations')


if __name__ == '__main__':
  merge(*sys.argv[1:])
