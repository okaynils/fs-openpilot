"""Parallel dataset generation: tracks x episodes (each worker owns one Panda3D instance)."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fsop.sim.raceline import PLANNER_VERSION


@dataclass
class EpisodeSpec:
  episode_id: str
  track_path: str  # json path, or "skidpad" / "acceleration"
  seed: int
  dr: str
  random_start: bool
  max_seconds: float
  noise: bool = True
  profile: str = ""
  distribution: str = ""
  physical_camera: bool = False
  configuration_seed: int = 0
  family: str = ""
  split: str = ""
  scenario: str = "drive"
  expert_mode: str = "random"
  offset_max: float = 0.0  # recovery excursions (NoiseCfg.offset_max)
  venue_prob: float = 0.0  # venue-realism scenes (DRCfg.venue_prob)


_RENDERER = None


def _worker(args):
  global _RENDERER
  specs, out_dir = args
  from fsop.profile import load_profile
  from fsop.sim.env import CAMERAS, FSEnv, SimCfg
  from fsop.sim.recorder import record_episode
  from fsop.sim.render.panda3d_backend import Panda3DRenderer
  from fsop.track.fixed_layouts import acceleration, skidpad
  from fsop.track.io import load_json

  rows = []
  for sp in specs:
    episode_started = time.monotonic()
    if sp.track_path == "skidpad":
      tr = skidpad()
    elif sp.track_path.startswith("acceleration"):
      tr = acceleration(width=float(sp.track_path.split(":")[1]) if ":" in sp.track_path else 3.0)
    else:
      tr = load_json(sp.track_path)
    profile = load_profile(sp.profile or None)
    provenance = None
    if sp.distribution:
      from fsop.sim.distribution import sample_profile
      profile, provenance = sample_profile(sp.distribution, sp.configuration_seed, sp.family or None)
    from fsop.geometry.camera import CameraModel
    cameras = ({c.name: CameraModel(c.name, c.K, c.width, c.height) for c in profile.cameras}
               if sp.distribution or sp.physical_camera else CAMERAS)
    signature = [(k, c.width, c.height, c.K.tolist()) for k, c in cameras.items()]
    if _RENDERER is not None and getattr(_RENDERER, "signature", None) != signature:
      _RENDERER.close()
      _RENDERER = None
    if _RENDERER is None:
      _RENDERER = Panda3DRenderer(cameras)
      _RENDERER.signature = signature
    renderer = _RENDERER
    cfg = SimCfg(physical_camera=bool(sp.distribution or sp.physical_camera), dr=sp.dr, random_start=sp.random_start,
                 venue_prob=sp.venue_prob)
    if sp.scenario == "stop":
      if tr.discipline != "acceleration":
        raise ValueError("Stop scenarios require an acceleration course")
      tr.start_pose = (tr.finish_line[0][0] + 8.0, 0.0, 0.0)
      cfg.start_speed = min(profile.vehicle.max_speed * .8, 6.0)
    elif sp.scenario != "drive":
      raise ValueError(f"Unknown scenario: {sp.scenario}")
    cfg.noise.enabled = sp.noise
    cfg.noise.offset_max = sp.offset_max
    # Distribution owns installation variation; do not add car-sized legacy jitter to rovers.
    from fsop.sim.randomization import dr_preset
    dr = dr_preset(sp.dr)
    if sp.distribution:
      dr.cam_height_jitter = dr.cam_pitch_deg = dr.cam_roll_deg = dr.cam_yaw_deg = 0.0
    expert_params = None
    if sp.distribution or sp.expert_mode == "configured":
      from fsop.sim.expert import configured_expert
      expert_params = configured_expert(profile.vehicle)
    setup_finished = time.monotonic()
    env = FSEnv(tr, cfg, seed=sp.seed, renderer=renderer, profile=profile, dr_cfg=dr, expert_params=expert_params)
    environment_finished = time.monotonic()
    meta = record_episode(env, out_dir, sp.episode_id, max_seconds=sp.max_seconds)
    meta["generation_timing"] = {"setup_seconds": setup_finished - episode_started,
                                 "environment_seconds": environment_finished - setup_finished,
                                 "record_seconds": time.monotonic() - environment_finished}
    Path(out_dir, sp.episode_id, "meta.json").write_text(json.dumps(meta, indent=1))
    if provenance:
      meta["distribution"] = provenance
      meta["split"] = sp.split
      Path(out_dir, sp.episode_id, "meta.json").write_text(json.dumps(meta, indent=1))
    rows.append({"id": sp.episode_id, "track_id": tr.id, "discipline": tr.discipline, "frames": meta["frames"],
                 "split": sp.split, "dr": sp.dr, "result": meta["summary"]["result"], "doo": meta["summary"]["doo"]})
  return rows


def plan_dataset(track_dir: str, n_random: int = 1, n_standing: float = 0.15, n_skidpad: int = 10,
                 n_accel: int = 10, max_seconds: float = 40.0, dr_mix=(("light", 0.6), ("heavy", 0.3), ("off", 0.1)),
                 seed: int = 0, profile: str = "", offset_max: float = 0.0,
                 venue_prob: float = 0.0) -> list[EpisodeSpec]:
  rng = np.random.default_rng(seed)
  idx = json.loads((Path(track_dir) / "index.json").read_text())["tracks"]
  levels, probs = zip(*dr_mix, strict=True)
  specs = []
  k = 0

  def dr():
    return str(rng.choice(levels, p=np.array(probs) / sum(probs)))

  for r in idx:
    path = str(Path(track_dir) / r["path"])
    for _ in range(n_random):
      specs.append(EpisodeSpec(f"ep{k:05d}_{r['id']}", path, seed + k, dr(), True, max_seconds))
      k += 1
    if rng.random() < n_standing:
      specs.append(EpisodeSpec(f"ep{k:05d}_{r['id']}_stand", path, seed + k, dr(), False, max_seconds))
      k += 1
  for i in range(n_skidpad):
    specs.append(EpisodeSpec(f"ep{k:05d}_skidpad", "skidpad", seed + k, dr(), False, 60.0, noise=i % 2 == 0))
    k += 1
  for i in range(n_accel):
    w = float(rng.uniform(3.0, 5.0))
    specs.append(EpisodeSpec(f"ep{k:05d}_accel", f"acceleration:{w:.2f}", seed + k, dr(), False, 30.0,
                             noise=i % 2 == 0))
    k += 1
  for sp in specs:
    sp.profile = profile
    sp.offset_max = offset_max if sp.track_path != "skidpad" else min(offset_max, 0.6)
    sp.venue_prob = venue_prob
  return specs


def record_dataset(specs: list[EpisodeSpec], out_dir: str, workers: int = 4) -> list[dict]:
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  # Resume only compatible labels. Otherwise an old dataset silently keeps its
  # centerline plans despite the new expert being selected by default.
  for spec in specs:
    metadata = out / spec.episode_id / "meta.json"
    if metadata.exists():
      planner = json.loads(metadata.read_text()).get("label_planner", {})
      if planner.get("version") != PLANNER_VERSION or planner.get("mode") != "racing":
        raise ValueError(f"{metadata}: incompatible expert labels; record into a fresh dataset directory")
  todo = [s for s in specs if not (out / s.episode_id / "meta.json").exists()]
  ctx = mp.get_context("spawn")
  rows = []
  # one task per episode keeps the progress bar honest; each worker process keeps its renderer
  with ctx.Pool(workers, maxtasksperchild=None) as pool:
    for r in tqdm(pool.imap_unordered(_worker, [([s], out_dir) for s in todo]), total=len(todo), desc="episodes"):
      rows.extend(r)
  index = []
  for s in specs:
    mp_ = out / s.episode_id / "meta.json"
    if mp_.exists():
      m = json.loads(mp_.read_text())
      index.append({"id": s.episode_id, "track_id": m["track_id"], "discipline": m["discipline"],
                    "frames": m["frames"], "dr": s.dr, "result": m["summary"]["result"]})
  (out / "index.json").write_text(json.dumps({"episodes": index}, indent=1))
  return index
