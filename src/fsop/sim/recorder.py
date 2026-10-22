"""Episode recording to the training dataset format.

data/sim/<dataset>/<episode_id>/
  meta.json         track id, discipline, seeds, scene DR, mount, vehicle + expert params, fps, cones
  track.json        copy of the track
  cones.npy         [M,3] episode cones actually rendered: x, y, class (after jitter / removal)
  frames.zarr/      road, wide : uint8 [T,6,128,256] (openpilot packed YUV), blosc-zstd chunks per frame
  states.npy        [T,12] t,x,y,yaw,vx,vy,r,steer,accel,s,lat,noise
  plan.npy          [T,33,15] counterfactual expert plan (ego frame, x fwd / y left / z up)
  desire.npy        [T,8] desire pulses
  cones_down.npy    [T] number of knocked cones so far (cheap) ; events.jsonl referee events
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from fsop.geometry.yuv import rgb_to_model
from fsop.sim.env import FSEnv, Obs
from fsop.track.io import save_json
from fsop.track.schema import CONE_CLASS_INDEX

STATE_COLS = ["t", "x", "y", "yaw", "vx", "vy", "r", "steer", "accel", "s", "lat", "noise"]


class EpisodeWriter:
  def __init__(self, out_dir: str | Path, env: FSEnv, episode_id: str, max_frames: int, record_every: int = 1):
    self.record_every = record_every
    self.dir = Path(out_dir) / episode_id
    self.dir.mkdir(parents=True, exist_ok=True)
    self.env = env
    self.id = episode_id
    self.T = 0
    store = zarr.storage.LocalStore(str(self.dir / "frames.zarr"))
    root = zarr.group(store=store, overwrite=True)
    comp = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
    self.arrays = {c: root.create_array(c, shape=(max_frames, 6, 128, 256), chunks=(1, 6, 128, 256), dtype="uint8",
                                        compressors=comp)
                   for c in env.cfg.cameras}
    self.states, self.plans, self.desires, self.down = [], [], [], []
    self.events = []

  def add(self, obs: Obs, plan: np.ndarray) -> None:
    for c, img in obs.images.items():
      self.arrays[c][self.T] = rgb_to_model(img)
    st = obs.state
    self.states.append([obs.t, st.x, st.y, st.yaw, st.vx, st.vy, st.r, st.steer, st.accel, obs.s, obs.lat, obs.noise])
    self.plans.append(plan.astype(np.float32))
    self.desires.append(obs.desire)
    self.down.append(int(self.env.referee.down.sum()))
    self.events.extend({"t": e.t, "kind": e.kind, **e.data} for e in obs.events)
    self.T += 1

  def close(self, summary: dict) -> dict:
    env = self.env
    for a in self.arrays.values():
      a.resize((self.T, 6, 128, 256))
    np.save(self.dir / "states.npy", np.asarray(self.states, np.float64))
    np.save(self.dir / "plan.npy", np.asarray(self.plans, np.float32))
    np.save(self.dir / "desire.npy", np.asarray(self.desires, np.float32))
    np.save(self.dir / "cones_down.npy", np.asarray(self.down, np.int32))
    cls = np.array([CONE_CLASS_INDEX.get(c, -1) for c in env.cone_colors], np.float64)
    np.save(self.dir / "cones.npy", np.c_[env.cones_xy, cls])
    save_json(env.track, self.dir / "track.json")
    with open(self.dir / "events.jsonl", "w") as f:
      for e in self.events:
        f.write(json.dumps(e, default=float) + "\n")
    m = env.mount
    meta = {
      "id": self.id, "track_id": env.track.id, "discipline": env.track.discipline, "seed": env.seed,
      "frames": self.T, "fps": env.cfg.fps / self.record_every, "sim_fps": env.cfg.fps,
      "record_every": self.record_every, "cameras": list(env.cfg.cameras), "state_cols": STATE_COLS,
      "mount": {"x": m.x, "y": m.y, "z": m.z, "roll": m.roll, "pitch": m.pitch, "yaw": m.yaw},
      "scene": env.scene.to_dict(), "vehicle": asdict(env.vp), "expert": asdict(env.expert_params),
      "dr": asdict(env.dr_cfg), "summary": summary, "schema_version": 1,
    }
    (self.dir / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def record_episode(env: FSEnv, out_dir: str | Path, episode_id: str, max_seconds: float | None = None,
                   record_every: int = 4) -> dict:
  """Drive with the expert (+ noise injection) and record frames with counterfactual labels.

  ``record_every=4`` stores the 5 Hz grid only: the model consumes frames t and t-4 (0.2 s apart) and a
  5 Hz feature history, so training samples on that grid need nothing in between (4x less storage).
  Rendering is skipped on frames that are not stored.
  """
  max_t = min(env.max_time, max_seconds) if max_seconds else env.max_time
  w = EpisodeWriter(out_dir, env, episode_id, max_frames=int(max_t * env.cfg.fps / record_every) + 2,
                    record_every=record_every)
  obs = env.observe()
  desire_acc = np.zeros(8, np.float32)
  while not env.done and obs.t < max_t:
    desire_acc = np.maximum(desire_acc, obs.desire)  # max-pool pulses over the skipped ticks (= modeld)
    if obs.frame % record_every == 0:
      obs.desire = desire_acc
      w.add(obs, env.label())
      desire_acc = np.zeros(8, np.float32)
    steer, accel = env.expert_action()
    render_next = (obs.frame + 1) % record_every == 0
    obs = env.step(steer, accel, render=render_next)
  meta = w.close(env.referee.summary())
  return meta
