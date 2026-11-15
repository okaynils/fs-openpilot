"""Summarize complete, paired CPU cases without treating line crossings as mission success."""
import json
import sys
from pathlib import Path

import numpy as np


def summarize(report):
  identity = report['identity']
  configs = identity['suite']['configurations']
  family = {c['name']: c['family'] for c in configs}
  expected = {(c['name'], 98011, 5101, bool(i % 2)) for i, c in enumerate(configs)}
  models = [*identity['checkpoints'], 'expert']
  actual_models = {r['checkpoint'] for r in report['results']}
  if actual_models != set(models):
    raise ValueError('Missing or unexpected models in paired CPU report')
  results = []
  for model in models:
    rows = [r for r in report['results'] if r['checkpoint'] == model]
    keys = [(r['configuration'], r['track_seed'], r['scene_seed'], r['recovery']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
      raise ValueError(f'Incomplete or duplicate paired cases: {model}')
    for group in ('all', 'formula', 'rover'):
      selected = [r for r in rows if group == 'all' or family[r['configuration']] == group]
      row = dict(checkpoint=model, family=group, cases=len(selected),
                 mission_passes=sum(r['pass'] for r in selected),
                 clean_mission_passes=sum(r['pass'] and r['doo'] == 0 and r['oc'] == 0 for r in selected),
                 contacts=sum(r['doo'] for r in selected), off_course=sum(r['oc'] for r in selected),
                 worst_clearance_m=min(r['minimum_clearance_m'] for r in selected))
      for key in ('completion_fraction', 'online_expert_ade_4s', 'one_second_tracking_error_m',
                  'rms_steer_rate_rad_s', 'policy_steady_p95_ms'):
        values = [r[key] for r in selected if r[key] is not None]
        row['mean_'+key] = float(np.mean(values)) if values else None
      results.append(row)
  failures = [dict(checkpoint=r['checkpoint'], configuration=r['configuration'], result=r['result'],
                   reported_lap_intervals=r['n_laps'], referee_complete=r['complete'], mission_pass=r['pass'])
              for r in report['results'] if not r['pass']]
  return dict(groups=results, failures=failures,
              limitations=['Within-run CPU comparison only; not the paired Modal collector ablation.',
                           'ADE and tracking error are measured on each policy\'s own visited states.',
                           'Referee line-crossing intervals alone do not prove mission completion.',
                           'Fixed-step simulation does not inject measured inference latency as a control delay.',
                           'No default replacement without the missing independent tests.'])


if __name__ == '__main__':
  report = json.loads(Path(sys.argv[1]).read_text())
  output = summarize(report)
  Path(sys.argv[2]).write_text(json.dumps(output, indent=2))
  print(json.dumps([r for r in output['groups'] if r['family'] == 'all'], indent=2))
