"""Parallel batch generation of track sets with an index for split/curriculum selection."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from fsop.track.generate import TrackGenCfg, generate_track
from fsop.track.io import load_json, save_json
from fsop.track.schema import Track


def _gen_one(args) -> dict:
  seed, cfg, out = args
  t = generate_track(seed, cfg)
  path = Path(out) / f"{t.id}.json"
  save_json(t, path)
  return {"id": t.id, "path": path.name, "seed": seed, **asdict(t.stats), "method": t.generator["method"],
          "slalom": t.generator.get("slalom") is not None}


def generate_set(out: str | Path, n: int, cfg: TrackGenCfg, seed0: int = 0, workers: int = 6) -> list[dict]:
  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  jobs = [(seed0 + i, cfg, out) for i in range(n)]
  with ProcessPoolExecutor(workers) as ex:
    rows = list(tqdm(ex.map(_gen_one, jobs, chunksize=4), total=n, desc="tracks"))
  with open(out / "index.json", "w") as f:
    json.dump({"cfg": asdict(cfg), "tracks": rows}, f, indent=1)
  return rows


def load_set(path: str | Path) -> list[Track]:
  path = Path(path)
  if path.is_file():
    return [load_json(path)]
  idx = json.loads((path / "index.json").read_text())
  return [load_json(path / r["path"]) for r in idx["tracks"]]


def split_ids(rows: list[dict], val_frac: float = 0.1, test_frac: float = 0.1) -> dict[str, list[str]]:
  """Deterministic split by seed (tracks, not episodes, are held out)."""
  out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
  for r in rows:
    h = (r["seed"] * 2654435761) % 1000 / 1000
    key = "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"
    out[key].append(r["id"])
  return out
