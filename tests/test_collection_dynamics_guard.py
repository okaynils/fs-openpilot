"""Collection intervention must recognize large high-speed curvature disagreement."""
import pytest

from fsop.sim.transfer_collection import InterventionCfg, takeover_deadline


def test_high_speed_departure_triggers_before_fixed_angle_limit():
  # Recorded train_v2_00_current departure at t=4.0: learner -0.0398,
  # expert -0.2014, speed 8.69 m/s, wheelbase 1.598 m.
  legacy = InterventionCfg(lateral_reference='reference_path')
  assert takeover_deadline(4., 0., .1617, 0., legacy) == 0.
  guarded = InterventionCfg(lateral_reference='reference_path', lateral_accel_disagreement_mps2=2.)
  assert takeover_deadline(4., 0., .1617, 0., guarded, speed=8.69, wheelbase=1.598,
                           steering=-.0398, expert_steering=-.2015) == 5.5
  assert takeover_deadline(4., 0., .1617, 0., guarded, speed=.5, wheelbase=1.598,
                           steering=-.0398, expert_steering=-.2015) == 0.


def test_dynamics_guard_requires_geometry_and_preserves_hysteresis():
  guarded = InterventionCfg(lateral_accel_disagreement_mps2=2.)
  with pytest.raises(ValueError, match='speed and vehicle geometry'):
    takeover_deadline(4., 0., 0., 0., guarded)
  assert takeover_deadline(4., 0., 0., 5., guarded, speed=2., wheelbase=.4,
                           steering=0., expert_steering=0.) == 5.
  with pytest.raises(ValueError):
    InterventionCfg(lateral_accel_disagreement_mps2=0.)
