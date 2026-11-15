"""Prevent unchanged initialization or a tied refresh from passing the pilot gate."""
import importlib
from pathlib import Path


def test_refresh_gate_rejects_tie_and_unchanged_winner(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
  module = importlib.import_module('transfer_v7_select')
  models = [module.PREVIOUS, module.CURRENT, module.INITIAL, module.BASELINE, module.V1, 'expert']
  report = {'results': [dict(checkpoint=m, configuration=str(i), track_seed=1, scene_seed=2,
                             recovery=False, **{'pass': True}, oc=0, doo=0, completion_fraction=1.)
                        for m in models for i in range(4)]}
  result = module.select(report)
  assert not result['pilot_pass'] and result['selected_for_further_evaluation'] is None
  for row in report['results']:
    if row['checkpoint'] in (module.PREVIOUS, module.INITIAL) and row['configuration'] == '0':
      row['pass'] = False
  assert module.select(report)['selected_for_further_evaluation'] == module.CURRENT
  for row in report['results']:
    if row['checkpoint'] == module.CURRENT and row['configuration'] == '1':
      row['pass'] = False
  assert module.select(report)['selected_for_further_evaluation'] is None
