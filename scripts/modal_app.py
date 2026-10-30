"""Run the fsop pipeline on Modal (https://modal.com): data generation fans out over CPU containers
(Panda3D renders headless through EGL + Mesa llvmpipe), caching / training / benchmarks run on a GPU.

Everything the pipeline writes lives on the Modal volume ``fsop`` and is linked into the container's
working copy, so the usual relative paths (``data/``, ``artifacts/``, ``runs/``) work unchanged.

  modal run scripts/modal_app.py::upload               # model files, tracks, v1 test split, init ckpt
  modal run scripts/modal_app.py::gen_data --name v2 --per-track 2 --offset-max 1.0
  modal run --detach scripts/modal_app.py::cache --dataset data/sim/v2 --ckpt runs/mac_stage2/ckpt/best.pt
  modal run --detach scripts/modal_app.py::train --cfg configs/train/cuda_stage2_v2.yaml
  modal run --detach scripts/modal_app.py::bench --name <id> --ckpt runs/<run>/ckpt/best.pt
  modal volume get fsop runs/<run>/ckpt/best.pt runs/<run>/ckpt/   # fetch results

GPU jobs are *spawned*: the entrypoint returns at once and the job keeps running however the local client
fares (with ``--detach``, a client-side error would otherwise still stop the app). Follow them with
``modal app logs <app id>`` or by reading their outputs on the volume.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
WORK = "/root/fsop"
VOL_DIR = "/vol"
LINKED = ("data", "artifacts", "runs")
TRAIN_GPU = os.environ.get("FSOP_GPU", "A100-40GB")

vol = modal.Volume.from_name("fsop", create_if_missing=True, version=2)
image = (
  modal.Image.debian_slim(python_version="3.12")
  .apt_install("git", "libegl1", "libgl1", "libgles2", "libgl1-mesa-dri", "libglib2.0-0")
  .uv_sync(str(REPO), extras=["sim"], frozen=True)
  .run_commands("python -m metadrive.pull_asset")
  .env({"PYTHONPATH": f"{WORK}/src", "PYTHONUNBUFFERED": "1"})
  .add_local_dir(REPO / "src", f"{WORK}/src", ignore=["**/__pycache__"])
  .add_local_dir(REPO / "configs", f"{WORK}/configs")
  .add_local_dir(REPO / "benchmarks", f"{WORK}/benchmarks")
  .add_local_file(REPO / "README.md", f"{WORK}/README.md")
)
app = modal.App("fsop", image=image, volumes={VOL_DIR: vol})


def _setup() -> None:
  for d in LINKED:
    Path(VOL_DIR, d).mkdir(parents=True, exist_ok=True)
    if not Path(WORK, d).exists():
      os.symlink(Path(VOL_DIR, d), Path(WORK, d))
  os.chdir(WORK)


def _sh(cmd: str) -> None:
  print("$", cmd, flush=True)
  subprocess.run(cmd, shell=True, check=True)


# ---------------------------------------------------------------------------------------------- data
@app.function(cpu=2.0, memory=4096, timeout=3 * 3600, max_containers=100)
def record_shard(specs: list[dict], out_dir: str) -> list[dict]:
  _setup()
  from fsop.sim.batch_record import EpisodeSpec, _worker

  rows = _worker(([EpisodeSpec(**s) for s in specs], out_dir))
  vol.commit()
  return rows


@app.function(cpu=8.0, memory=8192, timeout=6 * 3600)
def gen_dataset(name: str, track_sets: list[tuple[str, str, int, int]], per_track: int, offset_max: float,
                skidpad: int, accel: int, seconds: float, seed: int, shard: int, profile: str = "",
                venue_prob: float = 0.0) -> dict:
  """track_sets: (dir, preset yaml, n, seed) per set; generated on the volume if missing."""
  import json
  from dataclasses import asdict

  _setup()
  from fsop.sim.batch_record import plan_dataset

  specs = []
  for i, (tdir, preset, n, tseed) in enumerate(track_sets):
    if not Path(tdir, "index.json").exists():
      _sh(f"python -m fsop.cli track gen --out {tdir} --n {n} --seed {tseed} --cfg {preset} --workers 8")
    specs += plan_dataset(tdir, n_random=per_track, n_skidpad=skidpad if i == 0 else 0,
                          n_accel=accel if i == 0 else 0, max_seconds=seconds, seed=seed + 100_000 * i,
                          offset_max=offset_max, profile=profile, venue_prob=venue_prob)
  for i, sp in enumerate(specs):  # unique ids across track sets
    sp.episode_id = f"ep{i:05d}_" + sp.episode_id.split("_", 1)[1]
  out = f"data/sim/{name}"
  Path(out).mkdir(parents=True, exist_ok=True)
  vol.commit()
  done = {p.parent.name for p in Path(out).glob("*/meta.json")}
  todo = [asdict(s) for s in specs if s.episode_id not in done]
  print(f"{len(specs)} episodes planned, {len(todo)} to record", flush=True)
  chunks = [todo[i:i + shard] for i in range(0, len(todo), shard)]
  n_done = 0
  for rows in (record_shard.map(chunks, kwargs={"out_dir": out}, return_exceptions=True) if chunks else []):
    if isinstance(rows, Exception):
      print("shard failed:", rows, flush=True)
      continue
    n_done += len(rows)
    print(f"recorded {n_done}/{len(todo)}", flush=True)
  vol.reload()
  index = []
  for s in specs:
    mp = Path(out, s.episode_id, "meta.json")
    if mp.exists():
      m = json.loads(mp.read_text())
      index.append({"id": s.episode_id, "track_id": m["track_id"], "discipline": m["discipline"],
                    "frames": m["frames"], "dr": s.dr, "result": m["summary"]["result"]})
  Path(out, "index.json").write_text(json.dumps({"episodes": index}, indent=1))
  vol.commit()
  return {"episodes": len(index), "frames": sum(e["frames"] for e in index)}


@app.function(cpu=1.0, memory=2048, timeout=3600, max_containers=100)
def repack_shard(episode_dirs: list[str]) -> int:
  _setup()
  from fsop.sim.recorder import repack_frames

  n = sum(repack_frames(d) for d in episode_dirs)
  vol.commit()
  return n


@app.function(cpu=2.0, memory=4096, timeout=3 * 3600)
def repack_dataset(dataset: str, shard: int) -> int:
  """Convert a dataset recorded with one file per frame to sharded frames (network volumes are slow per file)."""
  import json

  _setup()
  ids = [e["id"] for e in json.loads(Path(dataset, "index.json").read_text())["episodes"]]
  chunks = [[f"{dataset}/{i}" for i in ids[k:k + shard]] for k in range(0, len(ids), shard)]
  n = 0
  for r in repack_shard.map(chunks, return_exceptions=True):
    if isinstance(r, Exception):
      print("shard failed:", r, flush=True)
    else:
      n += r
  print(f"repacked {n}/{len(ids)} episodes", flush=True)
  return n


# ---------------------------------------------------------------------------------------------- GPU
@app.function(gpu="L4", cpu=8.0, memory=32768, timeout=6 * 3600)
def cache_remote(dataset: str, ckpt: str | None, tag: str | None, hidden_only: bool) -> None:
  _setup()
  args = f"--dataset {dataset} --batch 64" + (f" --ckpt {ckpt}" if ckpt else "") + (f" --tag {tag}" if tag else "")
  _sh(f"python -m fsop.cli data cache-features {args}" + (" --hidden-only" if hidden_only else ""))
  vol.commit()


def _local_copy(src: Path, tag: str, stage: int) -> Path:
  """Copy a dataset to local disk: frames + labels + the ``tag`` cache (stage 2: hidden features only)."""
  import shutil
  import time
  from concurrent.futures import ThreadPoolExecutor

  dst = Path("/tmp/sim") / src.name
  skip = {"final.npy", "s16.npy"} if stage >= 2 else set()

  def ignore(d: str, names: list[str]) -> set[str]:
    return {n for n in names if n in skip or (n.startswith("cache_") and n != f"cache_{tag}")}

  t0 = time.time()
  dst.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src / "index.json", dst / "index.json")
  eps = [p for p in src.iterdir() if p.is_dir()]
  with ThreadPoolExecutor(32) as ex:
    list(ex.map(lambda p: shutil.copytree(p, dst / p.name, ignore=ignore, dirs_exist_ok=True), eps))
  print(f"copied {len(eps)} episodes to {dst} in {time.time() - t0:.0f} s", flush=True)
  return dst


@app.function(gpu=TRAIN_GPU, cpu=16.0, memory=65536, timeout=12 * 3600, ephemeral_disk=512 * 1024)
def train_remote(cfg: str, overrides: list[str], local_copy: bool) -> None:
  """``local_copy``: copy the dataset (frames + hidden cache) to local SSD first; random zarr reads from the
  network volume are the bottleneck otherwise."""
  _setup()
  from fsop.config import load_config
  from fsop.train.trainer import TrainCfg

  c = load_config(TrainCfg, Path(cfg), overrides)
  if local_copy:
    overrides = [*overrides, f"sim_root={_local_copy(Path(c.sim_root), c.cache_tag, c.stage)}"]
  try:
    _sh("python -m fsop.cli train --cfg " + shlex.quote(cfg) + "".join(f" -o {shlex.quote(o)}" for o in overrides))
  finally:
    vol.commit()


@app.function(gpu="L4", cpu=8.0, memory=32768, timeout=6 * 3600)
def bench_remote(cmd: str) -> None:
  _setup()
  try:
    _sh(cmd)
  finally:
    vol.commit()


# ---------------------------------------------------------------------------------------------- entrypoints
def _spawned(call) -> None:
  print(f"spawned {call.object_id}; follow with `modal app logs` (run with --detach so the app outlives this client)")

def _put(batch, local: Path, remote: str) -> None:
  if local.is_dir():
    batch.put_directory(str(local), remote)
  else:
    batch.put_file(str(local), remote)


@app.local_entrypoint()
def upload(init_ckpt: str = "runs/mac_stage2/ckpt/best.pt", v1_test: bool = True) -> None:
  """Model files, track sets, the init checkpoint and the v1 *test* split (frames only, for the benchmark)."""
  import json

  os.chdir(REPO)
  with vol.batch_upload(force=True) as b:
    for p in ("artifacts/models/prep", "artifacts/models/driving_supercombo_b361e952.onnx", "data/tracks/td_v1",
              "data/tracks/fixed", init_ckpt):
      if Path(p).exists():
        _put(b, Path(p), "/" + p)
    if v1_test:
      # the modal CLI's environment does not have fsop installed: ask the project's environment
      ids = json.loads(subprocess.run(
        ["uv", "run", "python", "-c", "import json; from fsop.data.sim_dataset import split_episodes; "
         "print(json.dumps(split_episodes('data/sim/v1', 'test')))"], check=True, capture_output=True,
        text=True).stdout)
      idx = json.loads(Path("data/sim/v1/index.json").read_text())["episodes"]
      Path("/tmp/fsop_v1_index.json").write_text(json.dumps({"episodes": [e for e in idx if e["id"] in ids]}))
      b.put_file("/tmp/fsop_v1_index.json", "/data/sim/v1/index.json")
      for i in ids:
        for f in Path("data/sim/v1", i).iterdir():
          if not f.name.startswith("cache_"):
            _put(b, f, f"/data/sim/v1/{i}/{f.name}")
  print("uploaded")


@app.local_entrypoint()
def gen_data(name: str = "v2", per_track: int = 2, offset_max: float = 1.0, n_td: int = 480, n_tight: int = 160,
             skidpad: int = 40, accel: int = 20, seconds: float = 40.0, seed: int = 0, shard: int = 6,
             tracks: str = "", profile: str = "", venue_prob: float = 0.0) -> None:
  """``tracks``: name of the track sets (default: ``name``), so several datasets can share tracks."""
  t = tracks or name
  sets = [(f"data/tracks/td_{t}", "configs/track/trackdrive.yaml", n_td, 20_000)]
  if n_tight:
    sets.append((f"data/tracks/tight_{t}", "configs/track/tight.yaml", n_tight, 30_000))
  _spawned(gen_dataset.spawn(name, sets, per_track, offset_max, skidpad, accel, seconds, seed, shard, profile,
                             venue_prob))


@app.local_entrypoint()
def repack(dataset: str = "data/sim/v2", shard: int = 8) -> None:
  print(repack_dataset.remote(dataset, shard))


@app.local_entrypoint()
def cache(dataset: str = "data/sim/v2", ckpt: str = "", tag: str = "", hidden_only: bool = True) -> None:
  _spawned(cache_remote.spawn(dataset, ckpt or None, tag or None, hidden_only))


@app.local_entrypoint()
def train(cfg: str, o: str = "", local_copy: bool = True) -> None:
  """``--o "steps=100 batch_size=8"``: space-separated dot-list overrides."""
  _spawned(train_remote.spawn(cfg, o.split(), local_copy))


@app.local_entrypoint()
def bench(name: str, ckpt: str, suite: str = "configs/eval/closed_loop_quick.yaml") -> None:
  """Standard benchmark: open loop on the v1 test split + closed-loop suite, written to runs/bench/<name>/
  (fetch it and record with ``fsop bench add --open-loop .../ol.json --closed-loop .../report.json``)."""
  q, c = shlex.quote, shlex.quote(ckpt)
  out = f"runs/bench/{q(name)}"
  _spawned(bench_remote.spawn(f"mkdir -p {out} && python -m fsop.cli eval open-loop --dataset data/sim/v1 --ckpt {c} "
                      f"--out {out}/ol.json && python -m fsop.cli eval closed-loop --ckpt {c} --suite {q(suite)} "
                      f"--out {out} --no-rrd"))


@app.local_entrypoint()
def sh(cmd: str) -> None:
  """Escape hatch: run any shell command in the working copy (GPU container)."""
  bench_remote.remote(cmd)
