"""Parallel dataset generation: tracks x episodes (each worker owns one Panda3D instance)."""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


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
  offset_max: float = 0.0  # recovery excursions (NoiseCfg.offset_max)


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

  if _RENDERER is None:
    _RENDERER = Panda3DRenderer(CAMERAS)
  renderer = _RENDERER
  rows = []
  for sp in specs:
    if sp.track_path == "skidpad":
      tr = skidpad()
    elif sp.track_path.startswith("acceleration"):
      tr = acceleration(width=float(sp.track_path.split(":")[1]) if ":" in sp.track_path else 3.0)
    else:
      tr = load_json(sp.track_path)
    cfg = SimCfg(dr=sp.dr, random_start=sp.random_start)
    cfg.noise.enabled = sp.noise
    cfg.noise.offset_max = sp.offset_max
    env = FSEnv(tr, cfg, seed=sp.seed, renderer=renderer, profile=load_profile(sp.profile or None))
    meta = record_episode(env, out_dir, sp.episode_id, max_seconds=sp.max_seconds)
    rows.append({"id": sp.episode_id, "track_id": tr.id, "discipline": tr.discipline, "frames": meta["frames"],
                 "dr": sp.dr, "result": meta["summary"]["result"], "doo": meta["summary"]["doo"]})
  return rows


def plan_dataset(track_dir: str, n_random: int = 1, n_standing: float = 0.15, n_skidpad: int = 10,
                 n_accel: int = 10, max_seconds: float = 40.0, dr_mix=(("light", 0.6), ("heavy", 0.3), ("off", 0.1)),
                 seed: int = 0, profile: str = "", offset_max: float = 0.0) -> list[EpisodeSpec]:
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
  return specs


def record_dataset(specs: list[EpisodeSpec], out_dir: str, workers: int = 4) -> list[dict]:
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
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
