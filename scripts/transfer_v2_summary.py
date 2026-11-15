"""Summarize paired driving reports without hiding failed or off-course runs."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(report):
  groups = defaultdict(list)
  for row in report['results']:
    groups[row['checkpoint'], row['regime']].append(row)
  output = []
  for (checkpoint, regime), rows in groups.items():
    record = {'checkpoint': checkpoint, 'regime': regime, 'cases': len(rows),
              'passes': sum(r['pass'] for r in rows), 'complete_laps': sum(r['complete'] for r in rows),
              'contacts': sum(r['doo'] for r in rows), 'off_course': sum(r['oc'] for r in rows),
              'worst_clearance_m': min(r['minimum_clearance_m'] for r in rows),
              'elapsed_worker_seconds': sum(r['elapsed_seconds'] for r in rows),
              'failure_results': dict((status, sum(r['result'] == status for r in rows))
                                      for status in sorted({r['result'] for r in rows}))}
    for field in ('completion_fraction', 'rms_lateral_m', 'rms_steer_rate_rad_s',
                  'max_yaw_rate_rad_s', 'steering_saturation_fraction', 'mean_visible_cones',
                  'online_expert_ade_4s', 'one_second_tracking_error_m', 'policy_steady_p95_ms'):
      values = [r[field] for r in rows if r[field] is not None]
      record['mean_' + field] = float(np.mean(values)) if values else None
    recovery = [r['recovery_lateral_4_to_6s_m'] for r in rows if r['recovery']
                and r['recovery_lateral_4_to_6s_m'] is not None]
    record['recovery_lateral_4_to_6s_m'] = float(np.mean(recovery)) if recovery else None
    output.append(record)
  return {'groups': output, 'note': 'Referee passes allow contacts. Progress can reach 1 without a valid lap.'}


if __name__ == '__main__':
  result = summarize(json.loads(Path(sys.argv[1]).read_text()))
  Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
  for r in result['groups']:
    print(r['checkpoint'], r['regime'], f"{r['passes']}/{r['cases']}",
          'contacts', r['contacts'], 'off_course', r['off_course'])
