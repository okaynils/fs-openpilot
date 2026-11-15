"""Recovery collection must permit model-visited states without losing takeover hysteresis."""
import pytest

from fsop.sim.transfer_collection import InterventionCfg, takeover_deadline


def test_disagreement_can_be_disabled_without_disabling_lateral_recovery():
  legacy = InterventionCfg()
  visited = InterventionCfg(steering_disagreement_rad=None)
  assert takeover_deadline(2., .1, .4, 0., legacy) == 3.5
  assert takeover_deadline(2., .1, .4, 0., visited) == 0.
  assert takeover_deadline(2., 1.3, .0, 0., visited) == 3.5
  assert takeover_deadline(2.2, .5, .4, 3.5, visited) == 3.5
  assert takeover_deadline(3., 1.3, .0, 3.5, visited) == 4.5


@pytest.mark.parametrize('kwargs', [{'lateral_limit_m': 0}, {'hold_seconds': -1},
                                   {'steering_disagreement_rad': float('nan')},
                                   {'lateral_reference': 'unsupported'}])
def test_invalid_collection_thresholds_are_rejected(kwargs):
  with pytest.raises(ValueError):
    InterventionCfg(**kwargs)


def test_valid_offset_racing_line_does_not_force_expert_takeover():
  from pathlib import Path

  import yaml

  from fsop.eval.transfer import transfer_track
  from fsop.profile import profile_from_dict
  from fsop.sim.env import FSEnv, SimCfg
  from fsop.sim.expert import configured_expert
  from fsop.sim.randomization import dr_preset
  from fsop.sim.transfer_collection import intervention_lateral_error

  entry = yaml.safe_load(Path('configs/distributions/transfer_v2_design.yaml').read_text())['configurations'][3]
  profile = profile_from_dict(entry['profile'])
  dr = dr_preset('light')
  dr.cam_height_jitter = dr.cam_pitch_deg = dr.cam_roll_deg = dr.cam_yaw_deg = 0.
  cfg = SimCfg(render=False, random_start=True, randomize_expert=False, dr='light')
  cfg.noise.enabled = False
  env = FSEnv(transfer_track(880004), cfg, 7104, profile=profile, dr_cfg=dr,
              expert_params=configured_expert(profile.vehicle))
  obs = env.observe()
  legacy = InterventionCfg(steering_disagreement_rad=None)
  reference = InterventionCfg(steering_disagreement_rad=None, lateral_reference='reference_path')
  assert abs(obs.lat) > 1.25  # real counterexample: a valid, deliberately offset racing line
  error = intervention_lateral_error(obs, env.mission, reference)
  assert abs(error) < .4
  assert takeover_deadline(obs.t, intervention_lateral_error(obs, env.mission, legacy), 0., 0., legacy) > 0
  assert takeover_deadline(obs.t, error, 0., 0., reference) == 0
