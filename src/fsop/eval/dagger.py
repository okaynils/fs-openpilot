"""DAgger data aggregation: the *model* drives, the expert labels every visited state.

Episodes are written in the normal dataset format (5 Hz grid, counterfactual expert plans), so they can
be appended to a dataset and used by the next training round (cache features, then stage 1/2).
With ``beta`` > 0 the expert takes over (mixed policy) with that probability per frame segment, which
keeps early, poor models from leaving the track immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fsop.eval.closed_loop import ModelPolicy, mission_brake
from fsop.sim.env import CAMERAS, FSEnv, SimCfg
from fsop.sim.recorder import EpisodeWriter


def dagger_round(ckpt: str | Path, tracks: list, out_dir: str | Path, seconds: float = 40.0, beta: float = 0.2,
                 dr: str = "light", seed: int = 0, record_every: int = 4) -> list[dict]:
  from fsop.sim.render.panda3d_backend import Panda3DRenderer

  out_dir = Path(out_dir)
  renderer = Panda3DRenderer(CAMERAS)
  policy = ModelPolicy.from_checkpoint(ckpt)
  rng = np.random.default_rng(seed)
  rows = []
  for i, track in enumerate(tracks):
    cfg = SimCfg(dr=dr, random_start=track.closed, max_time=seconds)
    cfg.noise.enabled = False
    env = FSEnv(track, cfg, seed=seed + i, renderer=renderer)
    policy.reset()
    eid = f"dagger_{Path(str(ckpt)).parent.parent.name}_{i:04d}_{track.id}"
    w = EpisodeWriter(out_dir, env, eid, max_frames=int(seconds * env.cfg.fps / record_every) + 2,
                      record_every=record_every)
    obs = env.observe()
    expert_until = -1.0
    desire_acc = np.zeros(8, np.float32)
    while not env.done and obs.t < seconds:
      steer, accel, _ = policy(obs, env)  # keeps the model's temporal state current
      if obs.t >= expert_until and rng.random() < beta / env.cfg.fps:
        expert_until = obs.t + rng.uniform(1.0, 3.0)
      if obs.t < expert_until or abs(obs.lat) > 1.6:  # expert takes over near the boundary
        steer, accel = env.expert_action()
      accel = mission_brake(accel, obs, env)
      desire_acc = np.maximum(desire_acc, obs.desire)
      if obs.frame % record_every == 0:
        obs.desire = desire_acc
        w.add(obs, env.label())
        desire_acc = np.zeros(8, np.float32)
      obs = env.step(steer, accel, add_noise=False)
    meta = w.close(env.referee.summary())
    rows.append({"id": eid, "track_id": track.id, "discipline": track.discipline, "frames": meta["frames"],
                 "dr": dr, "result": meta["summary"]["result"]})
  return rows


def append_to_index(dataset: str | Path, rows: list[dict]) -> None:
  p = Path(dataset) / "index.json"
  idx = json.loads(p.read_text()) if p.exists() else {"episodes": []}
  have = {e["id"] for e in idx["episodes"]}
  idx["episodes"] += [r for r in rows if r["id"] not in have]
  p.write_text(json.dumps(idx, indent=1))
