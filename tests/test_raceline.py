import numpy as np
import pytest

from fsop.sim.raceline import RacingLimits, optimize_raceline, velocity_profile
from fsop.sim.vehicle import VehicleParams
from fsop.track import centerline as cl
from fsop.track.schema import Track


def circle():
  angle = np.linspace(0, 2 * np.pi, 160, endpoint=False)
  return Track("circle", "trackdrive", 0, True, 20 * np.c_[np.cos(angle), np.sin(angle)],
               np.full(160, 2.5), np.full(160, 2.5))


def test_circle_improves_time_with_clearance():
  track = circle()
  original = track.centerline.copy()
  vehicle = VehicleParams()
  line = optimize_raceline(track, vehicle)
  assert line.optimized, line.solver_message
  assert line.travel_time < 0.98 * line.baseline_time
  assert line.offset.min() > 0.5  # shorter radius beats the centerline on this track
  assert np.max(line.offset) + vehicle.width / 2 + RacingLimits().clearance <= 2.5 + 1e-6
  np.testing.assert_array_equal(track.centerline, original)


def test_speed_obeys_combined_grip_and_segment_lengths():
  x = np.linspace(0, 100, 301)
  points = np.c_[x, 5 * np.sin(x / 9)]
  limits = RacingLimits(v_max=25)
  vehicle = VehicleParams(mu=0.65, max_accel=3, max_brake=4)
  speed = velocity_profile(points, False, limits, vehicle)
  kappa = np.abs(cl.curvature(points, False))
  ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
  accel = np.diff(speed**2) / (2 * ds)
  fraction = np.sqrt(1 - (speed**2 * kappa / (vehicle.mu * 9.81))**2)
  available = np.minimum(fraction[:-1], fraction[1:])
  assert np.all(speed**2 * kappa <= 0.85 * vehicle.mu * 9.81 + 1e-6)
  assert np.all(accel <= vehicle.max_accel * available + 1e-6)
  assert np.all(-accel <= vehicle.max_brake * available + 1e-6)
  steer = np.arctan(vehicle.wheelbase * cl.curvature(points, False))
  assert np.all(np.abs(np.diff(steer)) / ds * np.maximum(speed[:-1], speed[1:]) <=
                vehicle.max_steer_rate + 1e-6)


def test_failed_solve_reports_baseline(monkeypatch):
  from types import SimpleNamespace

  from fsop.sim import raceline

  def fail(fun, x, **kwargs):
    return SimpleNamespace(x=np.full_like(x, np.nan), success=False, message="failed", nit=1)

  monkeypatch.setattr(raceline, "minimize", fail)
  track = circle()
  line = optimize_raceline(track, VehicleParams())
  assert not line.optimized and not line.converged
  np.testing.assert_array_equal(line.points, track.centerline)
  assert line.travel_time == line.baseline_time


def test_budget_limited_feasible_improvement_is_retained(monkeypatch):
  from types import SimpleNamespace

  from fsop.sim import raceline

  def limited(fun, x, **kwargs):
    feasible = np.full_like(x, 0.5)
    kwargs["callback"](feasible)
    # The last iterate can be infeasible; retain the earlier checked solution.
    return SimpleNamespace(x=np.full_like(x, 5.0), success=False, message="Iteration limit reached", nit=2)

  monkeypatch.setattr(raceline, "minimize", limited)
  line = optimize_raceline(circle(), VehicleParams())
  assert line.optimized and not line.converged
  assert line.travel_time < line.baseline_time
  np.testing.assert_allclose(line.offset, 0.5)


def test_impossible_clearance_is_rejected():
  with pytest.raises(ValueError, match="too narrow"):
    optimize_raceline(circle(), VehicleParams(width=5))


def test_mission_keeps_route_progress_but_labels_follow_racing_reference():
  from fsop.sim.expert import Expert, ExpertParams, Mission, plan_from_rollout, rollout
  from fsop.sim.vehicle import VehicleState

  track = circle()
  vehicle = VehicleParams()
  racing = Mission(track, ExpertParams(), vehicle)
  baseline = Mission(track, ExpertParams(path_mode="centerline"), vehicle)
  state = VehicleState(x=20.0, yaw=np.pi / 2, vx=8.0)
  hint, station = racing.init_progress(state.x, state.y)
  assert (hint, station) == baseline.init_progress(state.x, state.y)
  plans = []
  for mission in (racing, baseline):
    trajectory = rollout(mission, Expert(mission, vehicle), vehicle, state, station, hint)
    plans.append(plan_from_rollout(trajectory))
  # The new path must actually reach the training target, not just a debug overlay.
  assert np.linalg.norm(plans[0][16, :2] - plans[1][16, :2]) > 0.5
  np.testing.assert_array_equal(racing.pts, track.centerline)


def test_recorded_reference_has_label_provenance(tmp_path):
  import json

  from fsop.sim.env import FSEnv, SimCfg
  from fsop.sim.recorder import EpisodeWriter
  from fsop.track.fixed_layouts import acceleration

  env = FSEnv(acceleration(), SimCfg(render=False, cameras=(), randomize_expert=False))
  writer = EpisodeWriter(tmp_path, env, "episode", 1)
  writer.add(env.observe(), env.label())
  writer.close(env.referee.summary())
  meta = json.loads((tmp_path / "episode/meta.json").read_text())
  assert meta["label_planner"]["mode"] == "racing"
  reference = np.load(tmp_path / "episode/reference.npz")
  np.testing.assert_allclose(reference["points"], env.mission.path)
  np.testing.assert_allclose(reference["speed"], env.mission.v_ref)


def test_record_resume_rejects_legacy_labels(tmp_path):
  from fsop.sim.batch_record import EpisodeSpec, record_dataset

  (tmp_path / "old").mkdir()
  (tmp_path / "old/meta.json").write_text('{"frames": 10}')
  spec = EpisodeSpec("old", "acceleration", 0, "off", False, 1.0)
  with pytest.raises(ValueError, match="fresh dataset"):
    record_dataset([spec], str(tmp_path))


def test_racing_line_avoids_actual_scene_cone():
  track = circle()  # nominal layout has no cones
  vehicle = VehicleParams()
  nominal = optimize_raceline(track, vehicle)
  scene = optimize_raceline(track, vehicle, cones_xy=np.array([[18.4, 0.0]]))
  assert nominal.points[0, 0] < 19.1
  assert scene.points[0, 0] > 19.5  # gives the shifted cone body clearance
  assert scene.optimized
  assert scene.clearance_required == RacingLimits().clearance
  assert scene.travel_time < scene.baseline_time


def test_infeasible_centerline_is_not_a_time_target(monkeypatch):
  from fsop.sim import raceline

  # Isolate path feasibility from the separate clearance-dependent speed cap.
  monkeypatch.setattr(raceline, "velocity_profile",
                      lambda points, closed, limits, vehicle, **kwargs: np.full(len(points), limits.v_max))
  x = np.linspace(0, 30, 121)
  track = Track("shifted_boundary", "acceleration", 0, False, np.c_[x, np.zeros_like(x)],
                np.full(len(x), 1.5), np.full(len(x), 1.5))
  line = optimize_raceline(track, VehicleParams(), cones_xy=np.array([[15.0, 1.0]]))
  assert not line.baseline_feasible
  assert line.clearance_required == RacingLimits().clearance_min
  assert line.optimized
  assert line.points[len(x) // 2, 1] < -0.04
  assert line.travel_time > line.baseline_time  # necessary detour beats an invalid straight path


def test_tight_clearance_requires_anticipatory_local_slowdown():
  points = np.c_[np.arange(101), np.zeros(101)]
  limits, vehicle = RacingLimits(), VehicleParams()
  clearance = np.full(len(points), limits.clearance)
  normal = velocity_profile(points, False, limits, vehicle, clearance=clearance)
  clearance[50] = limits.clearance_min
  tight = velocity_profile(points, False, limits, vehicle, clearance=clearance)
  assert tight[50] <= limits.v_max * limits.clearance_min / limits.clearance + 1e-6
  assert tight[49] < normal[49]  # braking starts before the narrow point
  assert tight[0] == normal[0] and tight[-1] == normal[-1]  # no blanket speed reduction
