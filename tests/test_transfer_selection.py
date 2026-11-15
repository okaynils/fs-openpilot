"""Closed-loop selection must not reward leaving the course or compare missing cases."""
import runpy

import pytest

rank = runpy.run_path('scripts/transfer_v2_select.py')['rank']


def test_selection_prefers_valid_completion_and_requires_pairing():
  models = ['expert', 'runs/cuda_racing_v6/ckpt/best.pt', 'runs/transfer_minimal_v1/ckpt/best.pt',
            'runs/transfer_recovery_v2/ckpt/step_000500.pt',
            'runs/transfer_recovery_v2/ckpt/step_001500.pt']
  rows = []
  for model in models:
    # The later checkpoint has zero contacts because it leaves the course.
    failed = model == models[-1]
    rows.append(dict(checkpoint=model, configuration='held_out', track_seed=1, scene_seed=2,
                     recovery=False, **{'pass': not failed}, oc=int(failed), doo=0 if failed else 1,
                     completion_fraction=.1 if failed else 1.))
  result = rank({'results': rows})
  assert result['selected_candidate'] == models[-2]
  assert not result['improves_original_baseline_rank']
  with pytest.raises(ValueError, match='duplicate'):
    rank({'results': rows + [rows[-1]]})
  rows[-1]['scene_seed'] = 9
  with pytest.raises(ValueError, match='Unpaired'):
    rank({'results': rows})
