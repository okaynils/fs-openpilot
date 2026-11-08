import hashlib
import json

import numpy as np
import pytest

from fsop.data.relabel import relabel_episode
from fsop.data.sim_dataset import Episode
from fsop.sim.env import FSEnv, SimCfg
from fsop.sim.expert import Expert, ExpertParams, Mission
from fsop.sim.recorder import EpisodeWriter, write_frames
from fsop.track.fixed_layouts import acceleration, skidpad
from fsop.track.io import load_json


def record_source(tmp_path, track):
  env = FSEnv(track, SimCfg(render=False, cameras=()), expert_params=ExpertParams(path_mode="centerline"))
  if track.discipline == "skidpad":
    # An interior lap point, where nearest-point projection cannot identify the lap.
    env.hint = len(track.centerline) // 2
    env.s = float(env.mission.s_arr[env.hint])
    env.state.x, env.state.y = track.centerline[env.hint]
    env.state.yaw = float(track.heading()[env.hint])
    env.state.vx = 4.0
  writer = EpisodeWriter(tmp_path, env, "source", 1)
  writer.add(env.observe(), env.label())
  writer.close(env.referee.summary())
  source = tmp_path / "source"
  image = np.full((1, 6, 128, 256), 17, np.uint8)
  write_frames(source / "frames.zarr", {"road": image})
  return source, env


def test_relabel_matches_live_teacher_and_keeps_source_unchanged(tmp_path):
  source, env = record_source(tmp_path, skidpad())
  before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
  destination = tmp_path / "racing"
  meta = relabel_episode(source, destination)
  env.mission = Mission(load_json(source / "track.json"), ExpertParams(), env.vp, cones_xy=env.cones_xy)
  env.expert = Expert(env.mission, env.vp)
  np.testing.assert_allclose(np.load(destination / "plan.npy")[0], env.label(), atol=1e-5)
  after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
  assert before == after
  assert meta["behavior_expert"]["path_mode"] == "centerline"
  assert meta["expert"]["path_mode"] == "racing"
  assert meta["label_planner"]["rollout_safety"]["frames"] == meta["frames"]
  episode = Episode(destination)
  assert np.all(episode.frame_pair("road", 0) == 17)
  assert episode.cache_dir("new").parent == destination
  assert not (destination / "frames.zarr").exists()
  assert relabel_episode(source, destination) == meta  # compatible resume


def test_relabel_rejects_changed_source_and_in_place_output(tmp_path):
  source, _ = record_source(tmp_path, acceleration())
  with pytest.raises(ValueError, match="separate"):
    relabel_episode(source, source)
  destination = tmp_path / "racing"
  relabel_episode(source, destination)
  metadata = json.loads((source / "meta.json").read_text())
  metadata["seed"] += 1
  (source / "meta.json").write_text(json.dumps(metadata))
  with pytest.raises(ValueError, match="Incompatible"):
    relabel_episode(source, destination)


def test_racing_recipe_refuses_legacy_targets_before_loading_model(tmp_path, monkeypatch):
  from types import SimpleNamespace

  from fsop.train import trainer

  legacy = SimpleNamespace(eps=[SimpleNamespace(path=tmp_path / "old", meta={})])
  monkeypatch.setattr(trainer, "SimDataset", lambda *args, **kwargs: legacy)
  config = trainer.TrainCfg(out_dir=str(tmp_path), label_mode="racing", device="cpu")
  with pytest.raises(ValueError, match="expected racing labels"):
    trainer.Trainer(config)


def test_racing_checkpoint_selection_tracks_path_error():
  from fsop.train.trainer import TrainCfg, Trainer

  trainer = Trainer.__new__(Trainer)
  trainer.cfg = TrainCfg(selection_metric="ade_4s")
  trainer.best = float("inf")
  saved = []
  trainer.save = saved.append
  trainer._select_checkpoint({"loss": 1.0, "ade_4s": 2.0})  # baseline
  trainer._select_checkpoint({"loss": 0.5, "ade_4s": 3.0})  # better cones, worse path
  assert len(saved) == 1 and trainer.best == 2.0
  trainer._select_checkpoint({"loss": 1.1, "ade_4s": 1.5})
  assert len(saved) == 2 and trainer.best == 1.5
  trainer._select_checkpoint({"loss": 0.0, "ade_4s": float("nan")})
  assert len(saved) == 2
