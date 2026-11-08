"""Cache held-out plan predictions, then compare them against regenerated racing labels.

Prediction caches bind checkpoint contents, state contents and image location so
baseline and candidate must see the same observations. Test tracks are never used
for model selection; use the validation split for that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fsop.data.sim_dataset import Episode, split_episodes
from fsop.eval.open_loop import plan_errors, run_episode_model
from fsop.model.build import build_base
from fsop.sim.expert import T_IDXS
from fsop.train.device import pick_device


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation(ep):
  return {"states_sha256": digest(ep.path / "states.npy"),
          "frames": str(Path(ep.meta.get("frames_source", ep.path / "frames.zarr")).resolve())}


def point_at_distance(plan, distance=10.0):
  """Interpolate geometry by travelled distance, independently of its timing.

  Short predictions clamp to their endpoint and report missing coverage, so
  stopping early cannot remove a difficult frame from the comparison.
  """
  xy = plan[:, :, :2]
  if not np.isfinite(xy).all():
    raise ValueError("Nonfinite predicted path")
  station = np.c_[np.zeros(len(xy)), np.cumsum(np.linalg.norm(np.diff(xy, axis=1), axis=2), axis=1)]
  points = np.array([[np.interp(distance, s, p[:, j]) for j in range(2)]
                     for s, p in zip(station, xy, strict=True)]).reshape(-1, 2)
  return points, station[:, -1] >= distance


def mean_or_none(arrays):
  values = np.concatenate(arrays)
  return float(values.mean()) if len(values) else None


def predict(dataset, checkpoint, out, split):
  if out.exists() and any(out.iterdir()):
    raise ValueError("Use an empty prediction directory to avoid mixing checkpoint outputs")
  device = pick_device()
  model = build_base(checkpoint).to(device).eval()
  checkpoint_hash = digest(checkpoint)
  ids = split_episodes(dataset, split)
  if not ids:
    raise ValueError(f"Empty {split} split")
  out.mkdir(parents=True, exist_ok=True)
  manifest = {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
              "dataset": str(dataset.resolve()), "split": split, "episodes": {}}
  for eid in tqdm(ids, desc="predict"):
    ep = Episode(dataset / eid)
    result = run_episode_model(model, ep, device, bev=False)
    np.savez(out / f"{eid}.npz", mu=result["mu"], std=result["std"])
    manifest["episodes"][eid] = observation(ep)
  (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def compare(dataset, baseline, candidate, out):
  manifests = {"baseline": json.loads((baseline / "manifest.json").read_text()),
               "candidate": json.loads((candidate / "manifest.json").read_text())}
  ids = list(manifests["baseline"]["episodes"])
  if set(ids) != set(manifests["candidate"]["episodes"]) or not ids:
    raise ValueError("Prediction sets must cover the same nonempty held-out split")
  if manifests["baseline"]["split"] != manifests["candidate"]["split"]:
    raise ValueError("Prediction splits differ")
  metrics = {key: {} for key in manifests}
  changed_errors = {key: [] for key in manifests}
  geometry_errors = {key: [] for key in manifests}
  changed_geometry_errors = {key: [] for key in manifests}
  geometry_coverage = {key: [] for key in manifests}
  geometry_deltas = []
  deltas = []
  for eid in ids:
    ep = Episode(dataset / eid)
    if ep.meta.get("label_planner", {}).get("mode") != "racing" or "relabel" not in ep.meta:
      raise ValueError(f"{eid}: expected regenerated racing labels")
    old = np.load(Path(ep.meta["relabel"]["source"]) / "plan.npy")
    i2 = int(np.argmin(np.abs(T_IDXS - 2.0)))
    # Exclude temporal warm-up, consistently with existing open-loop evaluation.
    change = np.abs(ep.plan[5:, i2, 1] - old[5:, i2, 1])
    deltas.append(change)
    target_point, target_covers = point_at_distance(ep.plan[5:])
    old_point, old_covers = point_at_distance(old[5:])
    valid_geometry = target_covers & old_covers
    geometry_change = np.linalg.norm(target_point - old_point, axis=1)
    geometry_deltas.append(geometry_change[valid_geometry])
    for name, directory in (("baseline", baseline), ("candidate", candidate)):
      if manifests[name]["episodes"][eid] != observation(ep):
        raise ValueError(f"{name}/{eid}: observations differ from evaluation dataset")
      with np.load(directory / f"{eid}.npz") as pred:
        errors = plan_errors(pred["mu"][5:], ep.plan[5:], pred["std"][5:])
        predicted_point, predicted_covers = point_at_distance(pred["mu"][5:])
        geometry_error = np.linalg.norm(predicted_point - target_point, axis=1)
        geometry_errors[name].append(geometry_error[valid_geometry])
        changed_geometry_errors[name].append(geometry_error[valid_geometry & (geometry_change > 0.2)])
        geometry_coverage[name].append(predicted_covers[valid_geometry])
        for key, value in errors.items():
          metrics[name].setdefault(key, []).append(value)
        changed_errors[name].append(errors["lat_2.0s"][change > 0.2])
  report = {"dataset": str(dataset), "split": manifests["baseline"]["split"], "episodes": ids,
            "checkpoints": {name: m["checkpoint_sha256"] for name, m in manifests.items()},
            "mean_target_change_lat_2s": float(np.concatenate(deltas).mean()),
            "geometry_frames_10m": sum(len(d) for d in geometry_deltas),
            "mean_target_change_path_10m": mean_or_none(geometry_deltas),
            "changed_path_frames_10m": int(sum(np.sum(d > 0.2) for d in geometry_deltas)),
            "changed_frames": int(sum(np.sum(d > 0.2) for d in deltas)), "metrics": {}}
  for name, values in metrics.items():
    summary = {key: float(np.concatenate(arrays).mean()) for key, arrays in values.items()}
    changed = np.concatenate(changed_errors[name])
    summary["lat_2s_on_changed_targets"] = float(changed.mean()) if len(changed) else None
    summary["path_error_10m"] = mean_or_none(geometry_errors[name])
    summary["path_error_10m_on_changed_paths"] = mean_or_none(changed_geometry_errors[name])
    summary["path_coverage_10m"] = mean_or_none(geometry_coverage[name])
    report["metrics"][name] = summary
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2))
  print(json.dumps(report["metrics"], indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  pred = sub.add_parser("predict")
  pred.add_argument("--dataset", type=Path, required=True)
  pred.add_argument("--checkpoint", type=Path, required=True)
  pred.add_argument("--out", type=Path, required=True)
  pred.add_argument("--split", choices=["val", "test"], default="test")
  comp = sub.add_parser("compare")
  comp.add_argument("--dataset", type=Path, required=True)
  comp.add_argument("--baseline", type=Path, required=True)
  comp.add_argument("--candidate", type=Path, required=True)
  comp.add_argument("--out", type=Path, required=True)
  args = vars(parser.parse_args())
  command = args.pop("command")
  (predict if command == "predict" else compare)(**args)


if __name__ == "__main__":
  main()
