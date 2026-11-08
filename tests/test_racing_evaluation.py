import json
import runpy
from pathlib import Path

import numpy as np
import pytest


def test_comparison_rewards_learning_changed_targets_and_rejects_other_observations(tmp_path):
  api = runpy.run_path(str(Path(__file__).parents[1] / "scripts/evaluate_racing.py"))
  source, dataset = tmp_path / "source", tmp_path / "racing"
  source.mkdir()
  episode = dataset / "ep"
  episode.mkdir(parents=True)
  old = np.zeros((8, 33, 15), np.float32)
  old[:, :, 0] = np.linspace(0, 20, 33)
  target = old.copy()
  target[:, :, 1] = 0.5
  np.save(source / "plan.npy", old)
  np.save(episode / "plan.npy", target)
  np.save(episode / "states.npy", np.zeros((8, 12)))
  np.save(episode / "desire.npy", np.zeros((8, 8)))
  np.save(episode / "cones.npy", np.empty((0, 3)))
  (episode / "meta.json").write_text(json.dumps({
    "frames": 8, "mount": {}, "frames_source": str(source / "frames.zarr"),
    "label_planner": {"mode": "racing"}, "relabel": {"source": str(source)},
  }))
  observation = api["observation"](api["Episode"](episode))
  for name, prediction in (("baseline", old), ("candidate", target)):
    directory = tmp_path / name
    directory.mkdir()
    np.savez(directory / "ep.npz", mu=prediction, std=np.ones_like(prediction))
    (directory / "manifest.json").write_text(json.dumps({
      "split": "test", "checkpoint_sha256": name, "episodes": {"ep": observation},
    }))
  output = tmp_path / "comparison.json"
  api["compare"](dataset, tmp_path / "baseline", tmp_path / "candidate", output)
  result = json.loads(output.read_text())
  assert result["changed_frames"] == 3  # first five frames excluded consistently
  assert result["metrics"]["baseline"]["lat_2s_on_changed_targets"] == 0.5
  assert result["metrics"]["candidate"]["lat_2s_on_changed_targets"] == 0.0
  assert result["changed_path_frames_10m"] == 3
  assert result["metrics"]["baseline"]["path_error_10m_on_changed_paths"] == 0.5
  assert result["metrics"]["candidate"]["path_error_10m_on_changed_paths"] == 0.0
  assert result["metrics"]["candidate"]["path_coverage_10m"] == 1.0
  np.save(episode / "states.npy", np.ones((8, 12)))
  with pytest.raises(ValueError, match="observations differ"):
    api["compare"](dataset, tmp_path / "baseline", tmp_path / "candidate", output)


def test_geometry_comparison_ignores_speed_but_does_not_reward_stopping():
  api = runpy.run_path(str(Path(__file__).parents[1] / "scripts/evaluate_racing.py"))
  plans = np.zeros((3, 33, 15))
  plans[0, :, 0] = np.linspace(0, 20, 33)
  plans[1, :, 0] = np.linspace(0, 12, 33)  # same path, slower schedule
  points, covers = api["point_at_distance"](plans)
  np.testing.assert_allclose(points, [[10, 0], [10, 0], [0, 0]])
  np.testing.assert_array_equal(covers, [True, True, False])
  assert np.linalg.norm(points[2] - points[0]) == 10  # stationary output remains an error


def test_clearance_audit_transforms_body_heading_and_detects_contacts():
  from dataclasses import asdict

  from fsop.sim.vehicle import VehicleParams

  api = runpy.run_path(str(Path(__file__).parents[1] / "scripts/audit_racing_labels.py"))
  states = np.array([[4.0, 3.0, np.pi / 2]])
  plan = np.zeros((1, 2, 15))
  plan[0, 1, 11] = -np.pi / 2
  cones = np.array([[4.0, 4.4, 0.0]])
  hit = api["contacts"](plan, states, {"x": 0, "y": 1, "yaw": 2}, cones, asdict(VehicleParams()))
  np.testing.assert_array_equal(hit, [[True, False]])
