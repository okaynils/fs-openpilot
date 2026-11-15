"""Summarize the sealed final comparison, preserving paired outcomes and denominators."""
import json
from pathlib import Path

import numpy as np
from transfer_v2_summary import summarize


def summarize_final(root=Path('benchmarks/transfer_v2')):
  driving = json.loads((root / 'final_driving.json').read_text())
  offline = json.loads((root / 'final_open.json').read_text())
  summary = summarize(driving)
  summary['offline'] = []
  for checkpoint in offline['checkpoint_sha256']:
    for regime in ('interpolation', 'extrapolation'):
      rows = [r for r in offline['rows'] if r['checkpoint'] == checkpoint and r['regime'] == regime]
      weights = [r['evaluated_frames'] for r in rows]
      summary['offline'].append(dict(checkpoint=checkpoint, regime=regime, episodes=len(rows),
                                     frames=sum(weights),
                                     ade_4s=float(np.average([r['ade_4s'] for r in rows], weights=weights)),
                                     predicted_contact_frame_rate=float(np.average(
                                         [r['predicted_contact_frame_rate'] for r in rows], weights=weights)),
                                     minimum_predicted_clearance_m=min(
                                         r['minimum_predicted_clearance_m'] for r in rows)))
  summary['paired_pass_changes'] = []
  candidate = 'runs/transfer_selected_v2/selected.pt'
  def keys(row):
    return (row['configuration'], row['track_seed'], row['scene_seed'], row['recovery'])
  got = {keys(r): r for r in driving['results'] if r['checkpoint'] == candidate}
  for reference in ('runs/cuda_racing_v6/ckpt/best.pt', 'runs/transfer_minimal_v1/ckpt/best.pt'):
    old = {keys(r): r for r in driving['results'] if r['checkpoint'] == reference}
    if old.keys() != got.keys():
      raise ValueError('Unpaired final cases')
    counts = {'both_pass': 0, 'both_fail': 0, 'candidate_only_pass': 0, 'reference_only_pass': 0}
    for key, row in got.items():
      a, b = bool(row['pass']), bool(old[key]['pass'])
      counts['both_pass' if a and b else 'both_fail' if not a and not b else
             'candidate_only_pass' if a else 'reference_only_pass'] += 1
    summary['paired_pass_changes'].append(dict(reference=reference, **counts))
  summary['limitations'] = ['Twelve cases are a small deterministic simulator benchmark, not a population guarantee.',
                            'Offline ADE is frame-weighted; driving summaries weight complete cases equally.',
                            'Predicted contact rate concerns planned footprints, not actual cone contacts.',
                            'Final results were not used for checkpoint selection or further tuning.']
  (root / 'final_summary.json').write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  summarize_final()
