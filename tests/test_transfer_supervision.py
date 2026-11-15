"""Bad counterfactual labels must be excluded without compressing physical time."""
import json

import numpy as np
import pytest

from fsop.data.sim_dataset import Episode, SimDataset
from fsop.profile import load_profile, profile_from_dict


def test_invalid_labels_do_not_remove_history_frames(tmp_path):
  ep = tmp_path / 'ep'
  ep.mkdir()
  meta = {'frames': 4, 'mount': {}, 'vehicle': {}}
  (ep / 'meta.json').write_text(json.dumps(meta))
  for name, data in [('states', np.zeros((4, 12))), ('plan', np.zeros((4, 33, 15))),
                     ('desire', np.zeros((4, 8))), ('cones', np.empty((0, 3))),
                     ('label_valid', np.array([True, False, True, True]))]:
    np.save(ep / f'{name}.npy', data)
  (ep / 'cache_test').mkdir()
  hidden = np.repeat(np.arange(4)[:, None], 512, axis=1).astype(np.float32)
  np.save(ep / 'cache_test/hidden.npy', hidden)
  ds = SimDataset(tmp_path, episodes=['ep'], cache_tag='test')
  assert ds.index == [(0, 2), (0, 3)]
  assert ds.history(0, 2)[-1, 0] == 1  # invalid label, but the image remains real history
  np.save(ep / 'label_valid.npy', np.ones(3, bool))
  with pytest.raises(ValueError, match='supervision mask'):
    Episode(ep)


def test_generic_profile_dictionary_roundtrip():
  original = load_profile('configs/profiles/generic_rover.yaml')
  copied = profile_from_dict(original.to_dict())
  assert copied.to_dict() == original.to_dict()


def test_recorder_keeps_contact_between_stored_frames(tmp_path):
  from fsop.sim.env import FSEnv, SimCfg
  from fsop.sim.recorder import EpisodeWriter
  from fsop.sim.referee import Event
  from fsop.track.fixed_layouts import acceleration

  env = FSEnv(acceleration(), SimCfg(render=False, dr='off', randomize_expert=False), 3)
  writer = EpisodeWriter(tmp_path, env, 'episode', 1, record_every=4)
  obs = env.observe()
  writer.add(obs, env.label())
  # A contact at an unrecorded 20 Hz tick is absent from every stored observation.
  env.referee.events.append(Event(t=.05, kind='doo', data={'cone': 0}))
  writer.close(env.referee.summary())
  event = json.loads((writer.dir / 'events.jsonl').read_text())
  assert event == {'t': .05, 'kind': 'doo', 'cone': 0}
