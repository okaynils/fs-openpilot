"""Check body/cone contact at sampled counterfactual plan poses.

This is a label-quality diagnostic, not a continuous swept-volume guarantee.
The 0.5--4 s window excludes the immediate recorded state and covers the primary
supervision horizon. Both teachers are checked against the recorded scene cones.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fsop.sim.expert import T_IDXS
from fsop.track.rules import RULES


def contacts(plan, states, columns, cones, vehicle):
  """Return [frames, plan samples] footprint contacts, using the referee rectangle."""
  if not len(cones):
    return np.zeros(plan.shape[:2], dtype=bool)
  yaw = states[:, columns["yaw"]]
  c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
  x = states[:, columns["x"], None] + c * plan[:, :, 0] - s * plan[:, :, 1]
  y = states[:, columns["y"], None] + s * plan[:, :, 0] + c * plan[:, :, 1]
  xy = np.stack([x, y], axis=-1).reshape(-1, 2)
  heading = (yaw[:, None] + plan[:, :, 11]).ravel()
  front = vehicle["lf"] + vehicle["front_overhang"]
  rear = -(vehicle["lr"] + vehicle["rear_overhang"])
  half_width = vehicle["width"] / 2
  cone_radius = np.where(cones[:, 2] == 3, RULES.large_cone.base / 2, RULES.small_cone.base / 2)
  radius = np.hypot(max(front, -rear), half_width) + cone_radius.max()
  nearby = cKDTree(cones[:, :2]).query_ball_point(xy, radius)
  rows = np.repeat(np.arange(len(xy)), [len(ids) for ids in nearby])
  hit = np.zeros(len(xy), bool)
  if len(rows):
    ids = np.concatenate([np.asarray(ids, dtype=int) for ids in nearby])
    delta = cones[ids, :2] - xy[rows]
    h = heading[rows]
    x = np.cos(h) * delta[:, 0] + np.sin(h) * delta[:, 1]
    y = -np.sin(h) * delta[:, 0] + np.cos(h) * delta[:, 1]
    dx = np.maximum(np.maximum(rear - x, x - front), 0)
    dy = np.maximum(np.abs(y) - half_width, 0)
    np.logical_or.at(hit, rows, np.hypot(dx, dy) < cone_radius[ids])
  return hit.reshape(plan.shape[:2])


def _audit_episode(path):
  meta = json.loads(path.read_text())
  states = np.load(path.parent / "states.npy")
  cones = np.load(path.parent / "cones.npy")
  columns = {name: i for i, name in enumerate(meta["state_cols"])}
  record = {"episode": meta["id"], "frames": len(states)}
  directories = {"racing": path.parent}
  if "relabel" in meta:
    directories["original"] = Path(meta["relabel"]["source"])
  for name, directory in directories.items():
    plan = np.load(directory / "plan.npy")[:, (T_IDXS >= 0.5) & (T_IDXS <= 4)]
    hit = contacts(plan, states, columns, cones, meta["vehicle"]).any(1)
    record[name + "_frames_with_contact"] = int(hit.sum())
    record[name + "_contact_frame_indices"] = np.flatnonzero(hit).tolist()
  return record


def audit(dataset, out, workers=8, episode_ids=None):
  # Each episode reads several independent files from the remote volume.
  # Ordered map keeps the report deterministic while overlapping those reads.
  if episode_ids is None and (dataset / "index.json").exists():
    episode_ids = [row["id"] for row in json.loads((dataset / "index.json").read_text())["episodes"]]
  if episode_ids is not None:
    if any(Path(eid).name != eid or eid in (".", "..") for eid in episode_ids):
      raise ValueError("Invalid episode ID")
    paths = [dataset / eid / "meta.json" for eid in sorted(episode_ids)]
  else:
    paths = sorted(dataset.glob("*/meta.json"))
  with ThreadPoolExecutor(max_workers=workers) as pool:
    records = list(pool.map(_audit_episode, paths))
  summary = {"episodes": len(records), "frames": sum(r["frames"] for r in records)}
  for name in ("racing", "original"):
    summary[name + "_frames_with_contact"] = sum(r.get(name + "_frames_with_contact", 0) for r in records)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps({"scope": __doc__, "summary": summary, "episodes": records}, indent=2))
  print(json.dumps(summary))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  audit(args.dataset, args.out)
