"""Apply the predeclared paired correction pilot gate to complete driving cases."""
import json
import sys
from pathlib import Path

from transfer_v2_select import rank

PREVIOUS = 'runs/transfer_refresh_previous_pilot_v7_retry1/ckpt/step_000300.pt'
CURRENT = 'runs/transfer_refresh_current_pilot_v7/ckpt/step_000300.pt'
INITIAL = 'runs/transfer_state_speed_v5/ckpt/step_000250.pt'
BASELINE = 'runs/cuda_racing_v6/ckpt/best.pt'
V1 = 'runs/transfer_minimal_v1/ckpt/best.pt'


def select(report):
  ranking = rank(report)['ranking']  # validates paired, complete case sets
  rows = {r['checkpoint']: r for r in ranking}
  def key(checkpoint):
    r = rows[checkpoint]
    return (-r['passes'], r['off_course'], r['contacts'], -r['mean_progress'])
  gate = dict(current_better_than_previous=key(CURRENT) < key(PREVIOUS),
              current_better_than_initial=key(CURRENT) < key(INITIAL),
              current_at_least_baseline=key(CURRENT) <= key(BASELINE),
              current_at_least_v1=key(CURRENT) <= key(V1),
              expert_all_pass=rows['expert']['passes'] == rows['expert']['cases'])
  return dict(ranking=ranking, gate=gate, pilot_pass=all(gate.values()),
              selected_for_further_evaluation=CURRENT if all(gate.values()) else None,
              limitation='Validation only; default retention and independent final tests still required.')


if __name__ == '__main__':
  result = select(json.loads(Path(sys.argv[1]).read_text()))
  Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))
