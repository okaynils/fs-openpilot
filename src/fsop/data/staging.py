"""Materialize a training dataset and its referenced frames on local scratch disk."""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def stage_dataset(source: Path, destination: Path, cache_tag: str, stage: int, workers: int = 16) -> Path:
  source, destination = Path(source).resolve(), Path(destination).resolve()
  if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
    raise ValueError("Staging directories must be separate")
  index = json.loads((source / "index.json").read_text())
  ids = [row["id"] for row in index["episodes"]]
  if any(Path(eid).name != eid or eid in (".", "..") for eid in ids):
    raise ValueError("Invalid episode ID")
  skip = {"final.npy", "s16.npy"} if stage >= 2 else set()

  def ignore(directory, names):
    return {name for name in names if name in skip or
            (name.startswith("cache_") and name != f"cache_{cache_tag}")}

  def copy_episode(eid):
    src, dst = source / eid, destination / eid
    meta = json.loads((src / "meta.json").read_text())
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
    if "frames_source" in meta:
      frames = Path(meta.pop("frames_source"))
      shutil.copytree(frames, dst / "frames.zarr", dirs_exist_ok=True)
      (dst / "meta.json").write_text(json.dumps(meta, indent=1))

  destination.mkdir(parents=True, exist_ok=True)
  with ThreadPoolExecutor(workers) as pool:
    list(pool.map(copy_episode, ids))
  # Publish only after every episode and external frame store is present.
  for name in ("manifest.json", "distribution.yaml"):
    if (source / name).exists():
      shutil.copy2(source / name, destination / name)
  shutil.copy2(source / "index.json", destination / "index.json")
  return destination
