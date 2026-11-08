"""Replay a simulated state where a clear reference produced an unsafe tracked trajectory."""
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fsop.sim.expert import Expert, ExpertParams, Mission, rollout
from fsop.sim.vehicle import VehicleParams, VehicleState
from fsop.track.schema import Track


def recovery_case(monkeypatch):
  from fsop.sim import raceline

  with np.load(Path(__file__).parent / 'fixtures/racing_recovery.npz') as f:
    meta = json.loads(str(f['metadata']))
    track = Track('recovery', 'trackdrive', 0, True, f['centerline'], f['width_left'], f['width_right'], laps=10)
    reference = SimpleNamespace(points=f['points'], speed=f['speed'])
    cones, row = f['cones'], f['state']
  monkeypatch.setattr(raceline, 'optimize_raceline', lambda *args, **kwargs: reference)
  p, vp = ExpertParams(**meta['expert']), VehicleParams(**meta['vehicle'])
  m = Mission(track, p, vp, cones_xy=cones[:, :2])
  cols = {n: i for i, n in enumerate(meta['state_cols'])}
  st = VehicleState(**{n: float(row[cols[n]]) for n in ('x', 'y', 'yaw', 'vx', 'vy', 'r', 'steer', 'accel', 't')})
  s = float(row[cols['s']])
  hint = int(np.searchsorted(m.s_arr, s % m.L, side='right') - 1)
  return m, vp, st, s, hint, cones, meta


def contact_count(trajectory, cones, vehicle):
  # Independent audit implementation checks every simulated pose, including t=0.
  audit = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/audit_racing_labels.py'))
  return int(audit['contacts'](np.zeros((len(trajectory), 1, 15)), trajectory,
                              {'x': 0, 'y': 1, 'yaw': 2}, cones, vehicle).sum())


def test_racing_rollout_corrects_tracking_collision(monkeypatch):
  m, vp, st, s, hint, cones, meta = recovery_case(monkeypatch)
  expert = Expert(m, vp)
  speed_before, state_before = m.v_ref.copy(), st.as_array().copy()
  trajectory = rollout(m, expert, vp, st, s, hint, horizon=4., dt=.025)
  assert expert.rollout_stats["adjusted_frames"] == 1
  assert expert.rollout_stats["unresolved_frames"] == 0
  np.testing.assert_array_equal(m.v_ref, speed_before)
  np.testing.assert_array_equal(st.as_array(), state_before)
  assert contact_count(trajectory, cones, meta['vehicle']) == 0
  # A collision-free stop is insufficient: preserve useful racing progress.
  from fsop.sim.expert import _rollout

  _, nominal_s = _rollout(m, Expert(m, vp), vp, st, s, hint, horizon=4., dt=.025)
  _, selected_s = _rollout(m, Expert(m, vp, expert.last_rollout["speed_scale"]),
                          vp, st, s, hint, horizon=4., dt=.025)
  assert selected_s - s > .9 * (nominal_s - s)


def test_clear_nominal_rollout_is_unchanged(monkeypatch):
  from fsop.sim.expert import _rollout

  m, vp, st, s, hint, _, _ = recovery_case(monkeypatch)
  m.cone_tree = None
  expert = Expert(m, vp)
  nominal, _ = _rollout(m, expert, vp, st, s, hint, horizon=4., dt=.025)
  checked = rollout(m, expert, vp, st, s, hint, horizon=4., dt=.025)
  np.testing.assert_array_equal(checked, nominal)
  assert expert.last_rollout['speed_scale'] == 1.0
  assert expert.rollout_stats == {'frames': 1, 'adjusted_frames': 0, 'unresolved_frames': 0}


def test_initial_contact_is_reported_as_unresolved(monkeypatch):
  from scipy.spatial import cKDTree

  m, vp, st, s, hint, _, _ = recovery_case(monkeypatch)
  m.cones_xy = np.array([[st.x, st.y]])
  m.cone_tree = cKDTree(m.cones_xy)
  expert = Expert(m, vp)
  rollout(m, expert, vp, st, s, hint, horizon=.1, dt=.025)
  assert expert.last_rollout['contact_samples'] > 0
  assert expert.rollout_stats['unresolved_frames'] == 1
