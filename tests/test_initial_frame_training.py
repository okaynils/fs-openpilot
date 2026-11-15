"""An opt-in training contract includes real reset observations without inventing history."""
import json

import numpy as np
import pytest

from fsop.data.sim_dataset import SimDataset
from fsop.train.trainer import TrainCfg, Trainer


@pytest.mark.parametrize('include,expected', [(False, 1), (True, 0)])
def test_trainer_forwards_initial_frame_contract(tmp_path, monkeypatch, include, expected):
  class Observed(Exception):
    pass

  def dataset(*args, **kwargs):
    assert kwargs['min_k'] == expected
    raise Observed

  monkeypatch.setattr('fsop.train.trainer.SimDataset', dataset)
  cfg = TrainCfg(name='initial', out_dir=str(tmp_path), device='cpu', include_initial_frame=include)
  with pytest.raises(Observed):
    Trainer(cfg)


@pytest.mark.parametrize('mode', ['cached', 'fresh'])
def test_initial_frame_preserves_measured_speed_and_empty_history(tmp_path, mode):
  ep = tmp_path/'ep'
  ep.mkdir()
  (ep/'meta.json').write_text(json.dumps({'frames': 2, 'mount': {}, 'vehicle': {}}))
  states = np.zeros((2, 12))
  states[:, 4] = [1.5, 1.6]
  for name, data in [('states', states), ('plan', np.zeros((2, 33, 15))),
                     ('desire', np.zeros((2, 8))), ('cones', np.empty((0, 3)))]:
    np.save(ep/f'{name}.npy', data)
  cache = ep/'cache_test'
  cache.mkdir()
  for name, shape in [('hidden', (2, 512)), ('final', (2, 1024, 4, 8)), ('s16', (2, 256, 8, 16))]:
    np.save(cache/f'{name}.npy', np.full(shape, .25, np.float32))
  dataset = SimDataset(tmp_path, episodes=['ep'], min_k=0, cache_tag='test', image_targets_on=False,
                       mode='images' if mode == 'fresh' else 'cached', history_mode=mode)
  frames = np.full((2, 6, 2, 3), 120, np.uint8)
  dataset.eps[0].frames = {'road': frames, 'wide': frames}
  sample = dataset[0]
  assert sample['k'] == 0 and sample['speed'] == 1.5
  assert np.count_nonzero(sample['history']) == 0
  if mode == 'fresh':
    assert not sample['history_valid'].any()
    np.testing.assert_array_equal(sample['img'][:6], sample['img'][6:])
  else:
    np.testing.assert_array_equal(sample['hidden'], np.full(512, .25, np.float32))
