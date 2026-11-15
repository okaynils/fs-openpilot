"""Rank fixed closed-loop validation snapshots, refusing incomparable case sets."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rank(report):
  groups = {}
  for row in report['results']:
    groups.setdefault(row['checkpoint'], []).append(row)
  expected = {(r['configuration'], r['track_seed'], r['scene_seed'], r['recovery'])
              for r in groups['expert']}
  rows = []
  for checkpoint, cases in groups.items():
    keys = [(r['configuration'], r['track_seed'], r['scene_seed'], r['recovery']) for r in cases]
    if len(keys) != len(set(keys)) or set(keys) != expected:
      raise ValueError(f'Unpaired or duplicate validation cases: {checkpoint}')
    rows.append({'checkpoint': checkpoint, 'cases': len(cases),
                 'passes': sum(r['pass'] for r in cases), 'off_course': sum(r['oc'] for r in cases),
                 'contacts': sum(r['doo'] for r in cases),
                 'mean_progress': sum(r['completion_fraction'] for r in cases) / len(cases)})
  def key(row):
    return (-row['passes'], row['off_course'], row['contacts'], -row['mean_progress'])
  rows.sort(key=key)
  candidates = [r for r in rows if '/transfer_' in r['checkpoint'] and '/step_' in r['checkpoint']]
  if not candidates:
    raise ValueError('No named candidate snapshots in validation report')
  baseline = next(r for r in rows if 'cuda_racing_v6' in r['checkpoint'])
  previous = next(r for r in rows if 'transfer_minimal_v1' in r['checkpoint'])
  expert = next(r for r in rows if r['checkpoint'] == 'expert')
  return {'ranking': rows, 'selected_candidate': candidates[0]['checkpoint'],
          'improves_original_baseline_rank': key(candidates[0]) < key(baseline),
          'improves_previous_candidate_rank': key(candidates[0]) < key(previous),
          'expert_all_pass': expert['passes'] == expert['cases'],
          'promotion_requires_default_retention_and_independent_final_test': True}


if __name__ == '__main__':
  report = json.loads(Path(sys.argv[1]).read_text())
  result = rank(report)
  Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))
