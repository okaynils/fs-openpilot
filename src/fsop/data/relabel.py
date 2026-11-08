"""Regenerate counterfactual plans without rendering or changing source episodes.

Images are referenced read-only; states, geometry and other small arrays are copied.
Feature caches belong to the destination and must be rebuilt for its training model.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tqdm import tqdm

from fsop.sim.expert import Expert, ExpertParams, Mission, plan_from_rollout, rollout
from fsop.sim.raceline import PLANNER_VERSION
from fsop.sim.vehicle import VehicleParams, VehicleState
from fsop.track.io import load_json


def _fingerprint(source: Path, dt: float) -> str:
  digest = hashlib.sha256(str(source.resolve()).encode())
  digest.update(str(dt).encode())
  for name in ("meta.json", "track.json", "states.npy", "cones.npy"):
    digest.update((source / name).read_bytes())
  package = Path(__file__).parents[1]
  for name in ("data/relabel.py", "sim/expert.py", "sim/raceline.py", "sim/vehicle.py", "track/centerline.py"):
    digest.update((package / name).read_bytes())
  return digest.hexdigest()


def relabel_episode(source: Path, destination: Path, dt: float = 0.025) -> dict:
  source, destination = Path(source).resolve(), Path(destination).resolve()
  if destination == source or destination.is_relative_to(source):
    raise ValueError("Relabel destination must be separate from the source episode")
  if dt <= 0 or not np.isclose(0.05 / dt, round(0.05 / dt)):
    raise ValueError("Label dt must divide the 0.05 second control interval")
  fingerprint = _fingerprint(source, dt)
  if destination.exists():
    existing = destination / "meta.json"
    if existing.exists():
      meta = json.loads(existing.read_text())
      if meta.get("relabel", {}).get("fingerprint") == fingerprint:
        return meta
    raise ValueError(f"Incompatible destination episode: {destination}; use a fresh output directory")
  meta = json.loads((source / "meta.json").read_text())
  states = np.load(source / "states.npy")
  cols = {name: i for i, name in enumerate(meta["state_cols"])}
  names = ("x", "y", "yaw", "vx", "vy", "r", "steer", "accel", "t")
  if not set((*names, "s")).issubset(cols) or len(states) != meta["frames"]:
    raise ValueError(f"Invalid recorded state columns or frame count: {source}")
  if not np.isfinite(states).all():
    raise ValueError(f"Nonfinite recorded states: {source}")
  track = load_json(source / "track.json")
  vehicle = VehicleParams(**meta["vehicle"])
  params = ExpertParams(**{**meta["expert"], "path_mode": "racing"})
  scene_cones = np.load(source / "cones.npy")[:, :2]
  mission = Mission(track, params, vehicle, cones_xy=scene_cones)
  expert = Expert(mission, vehicle)
  plans = np.empty((len(states), 33, 15), np.float32)
  for i, row in enumerate(states):
    state = VehicleState(**{name: float(row[cols[name]]) for name in names})
    station = float(row[cols["s"]])
    # Stored route station disambiguates self-intersecting skidpad loops. A
    # global nearest-point query would jump to the wrong lap at the crossing.
    local = station % mission.L if mission.closed else station
    hint = int(np.clip(np.searchsorted(mission.s_arr, local, side="right") - 1, 0, mission.n - 1))
    traj = rollout(mission, expert, vehicle, state, station, hint, dt=dt)
    plans[i] = plan_from_rollout(traj, dt=dt)
  if not np.isfinite(plans).all():
    raise ValueError(f"Nonfinite regenerated plans: {source}")
  reference = mission.racing_line
  meta["behavior_expert"] = meta.get("behavior_expert", meta["expert"])
  meta["expert"] = asdict(params)
  meta["label_dt"] = dt
  meta["frames_source"] = str(Path(meta.get("frames_source", source / "frames.zarr")).resolve())
  if not Path(meta["frames_source"]).is_dir():
    raise ValueError(f"Missing source frames: {meta['frames_source']}")
  meta["relabel"] = {"source": str(source), "fingerprint": fingerprint}
  meta["label_planner"] = {
    "version": PLANNER_VERSION, "mode": "racing", "optimized": reference.optimized,
    "travel_time": reference.travel_time, "baseline_time": reference.baseline_time,
    "baseline_feasible": reference.baseline_feasible, "solver_message": reference.solver_message,
    "converged": reference.converged, "iterations": reference.iterations,
    "clearance_required": reference.clearance_required, "rollout_safety": expert.rollout_stats,
  }
  destination.parent.mkdir(parents=True, exist_ok=True)
  with TemporaryDirectory(prefix=".relabel-", dir=destination.parent) as tmp:
    pending = Path(tmp) / "episode"
    pending.mkdir()
    for name in ("track.json", "states.npy", "cones.npy", "desire.npy", "cones_down.npy", "events.jsonl"):
      if (source / name).exists():
        shutil.copy2(source / name, pending / name)
    np.save(pending / "plan.npy", plans)
    np.savez(pending / "reference.npz", route_s=mission.s_arr, points=reference.points,
             speed=reference.speed, offset=reference.offset)
    (pending / "meta.json").write_text(json.dumps(meta, indent=1))
    pending.rename(destination)
  return meta


def _worker(args):
  return relabel_episode(*args)


def relabel_dataset(source: str | Path, destination: str | Path, workers: int = 2,
                    dt: float = 0.025) -> dict:
  source, destination = Path(source).resolve(), Path(destination).resolve()
  if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
    raise ValueError("Source and destination dataset directories must be separate")
  index = json.loads((source / "index.json").read_text())
  jobs = []
  for row in index["episodes"]:
    episode_id = row["id"]
    if Path(episode_id).name != episode_id or episode_id in (".", ".."):
      raise ValueError(f"Invalid episode ID: {episode_id}")
    jobs.append((source / episode_id, destination / episode_id, dt))
  if not jobs or workers < 1:
    raise ValueError("Relabeling needs episodes and at least one worker")
  destination.mkdir(parents=True, exist_ok=True)
  with ProcessPoolExecutor(max_workers=workers) as pool:
    for _ in tqdm(pool.map(_worker, jobs), total=len(jobs), desc="relabel"):
      pass
  # The index is published only after every episode has committed successfully.
  pending = destination / "index.json.tmp"
  pending.write_text(json.dumps(index, indent=1))
  pending.replace(destination / "index.json")
  return {"source": str(source), "destination": str(destination), "episodes": len(jobs)}
