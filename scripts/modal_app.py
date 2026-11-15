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
  .add_local_dir(REPO / "tests", f"{WORK}/tests", ignore=["**/__pycache__"])
  .add_local_dir(REPO / "scripts", f"{WORK}/scripts", ignore=["**/__pycache__"])
  .add_local_dir(REPO / "benchmarks", f"{WORK}/benchmarks")
  .add_local_file(REPO / "README.md", f"{WORK}/README.md")
  .add_local_file(REPO / "pyproject.toml", f"{WORK}/pyproject.toml")
  .add_local_file(REPO / "uv.lock", f"{WORK}/uv.lock")
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


def _source_identity() -> dict:
  import hashlib

  files = [*Path("src/fsop").rglob("*.py"), *Path("configs").rglob("*.yaml"),
           *Path("scripts").rglob("*.py")]
  files += [p for p in (Path("pyproject.toml"), Path("uv.lock")) if p.exists()]
  return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def _archive_source(out: Path) -> None:
  import tarfile

  if not (out / "source.tar.gz").exists():
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
      for folder in ("src", "configs", "scripts", "tests", "pyproject.toml", "uv.lock"):
        if not Path(folder).exists():
          continue
        archive.add(folder, filter=lambda info: None if "__pycache__" in info.name else info)


# ---------------------------------------------------------------------------------------------- data
@app.function(cpu=2.0, memory=4096, timeout=3 * 3600, max_containers=100)
def record_shard(specs: list[dict], out_dir: str) -> list[dict]:
  _setup()
  from fsop.sim.batch_record import EpisodeSpec, _worker

  rows = _worker(([EpisodeSpec(**s) for s in specs], out_dir))
  vol.commit()
  return rows


@app.function(gpu=["L4", "A10"], cpu=4.0, memory=16384, timeout=3600, max_containers=3)
def transfer_record_shard(specs: list[dict], out_dir: str) -> list[dict]:
  """Bounded GPU rendering for low cameras where software rasterization is very slow."""
  _setup()
  from fsop.sim.batch_record import EpisodeSpec, _worker

  try:
    return _worker(([EpisodeSpec(**s) for s in specs], out_dir))
  finally:
    vol.commit()


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


# ---------------------------------------------------------------------------------------------- racing labels
@app.function(cpu=2.0, memory=4096, timeout=3 * 3600, max_containers=32)
def relabel_shard(source: str, destination: str, ids: list[str]) -> list[dict]:
  _setup()
  from fsop.data.relabel import relabel_episode

  results = []
  for eid in ids:
    try:
      meta = relabel_episode(Path(source) / eid, Path(destination) / eid)
      results.append({"id": eid, "frames": meta["frames"]})
    except Exception as exc:
      results.append({"id": eid, "error": str(exc)})
    finally:
      vol.commit()
  return results


@app.function(cpu=4.0, memory=8192, timeout=6 * 3600)
def relabel_remote(source: str, destination: str, shard: int = 6) -> dict:
  import json
  import runpy

  _setup()
  src, dst = Path(source).resolve(), Path(destination).resolve()
  if src == dst or src.is_relative_to(dst) or dst.is_relative_to(src):
    raise ValueError("Relabel into a separate dataset")
  index = json.loads(Path(source, "index.json").read_text())
  ids = [row["id"] for row in index["episodes"]]
  if any(Path(eid).name != eid or eid in (".", "..") for eid in ids):
    raise ValueError("Invalid episode ID")
  if not ids or shard < 1:
    raise ValueError("Relabeling requires a nonempty dataset and positive shard size")
  chunks = [ids[i:i + shard] for i in range(0, len(ids), shard)]
  output = Path(destination)
  output.mkdir(parents=True, exist_ok=True)
  vol.commit()
  errors, completed = [], 0
  for rows in relabel_shard.map([source] * len(chunks), [destination] * len(chunks), chunks,
                                return_exceptions=True):
    if isinstance(rows, Exception):
      errors.append({"error": str(rows)})
    else:
      errors.extend(row for row in rows if "error" in row)
      completed += sum("error" not in row for row in rows)
    print(f"relabeled {completed}/{len(ids)}, errors {len(errors)}", flush=True)
  vol.reload()
  (output / "relabel_errors.json").write_text(json.dumps(errors, indent=2))
  vol.commit()
  if errors or completed != len(ids):
    raise RuntimeError(f"Relabel incomplete: {completed}/{len(ids)}, {len(errors)} failures; see relabel_errors.json")
  quality_path = output / "label_quality.json"
  runpy.run_path(f"{WORK}/scripts/audit_racing_labels.py")["audit"](output, quality_path, episode_ids=ids)
  quality = json.loads(quality_path.read_text())["summary"]
  vol.commit()
  if quality["racing_frames_with_contact"] > quality["original_frames_with_contact"]:
    raise RuntimeError(f"Sampled label-contact regression; training withheld: {quality}")
  (output / "index.json").write_text(json.dumps(index, indent=1))
  vol.commit()
  return {"dataset": destination, "quality": quality}


@app.function(cpu=1.0, memory=2048, timeout=12 * 3600)
def racing_remote(source: str, destination: str, cfg: str) -> dict:
  """CPU relabel/quality gate -> GPU history cache -> GPU training. No local training."""
  from fsop.config import load_config
  from fsop.train.trainer import TrainCfg

  _setup()
  config = load_config(TrainCfg, Path(cfg))
  if config.sim_root != destination or config.label_mode != "racing":
    raise ValueError("Training configuration must require the relabeled racing dataset")
  if Path(config.out_dir, config.name).exists():
    raise ValueError("Refusing to overwrite an existing training run")
  labels = relabel_remote.remote(source, destination)
  cache_remote.remote(destination, config.init_ckpt, config.cache_tag, True)
  train_remote.remote(cfg, [], True)
  vol.reload()
  return {"labels": labels, "checkpoint": str(Path(config.out_dir, config.name, "ckpt/best.pt"))}


# ---------------------------------------------------------------------------------------------- GPU
@app.function(gpu=["L4", "A10"], cpu=8.0, memory=32768, timeout=6 * 3600)
def cache_remote(dataset: str, ckpt: str | None, tag: str | None, hidden_only: bool) -> None:
  import json
  import time

  _setup()
  started = time.time()
  args = f"--dataset {dataset} --batch 64" + (f" --ckpt {ckpt}" if ckpt else "") + (f" --tag {tag}" if tag else "")
  _sh(f"python -m fsop.cli data cache-features {args}" + (" --hidden-only" if hidden_only else ""))
  Path(dataset, f"cache_{tag}_timing.json").write_text(json.dumps({"seconds": time.time() - started,
                                                               "checkpoint": ckpt}, indent=2))
  vol.commit()


def _local_copy(src: Path, tag: str, stage: int) -> Path:
  """Copy a dataset to local disk: frames + labels + the ``tag`` cache (stage 2: hidden features only)."""
  import time

  from fsop.data.staging import stage_dataset

  t0 = time.time()
  dst = stage_dataset(src, Path("/tmp/sim") / src.name, tag, stage)
  print(f"staged dataset at {dst} in {time.time() - t0:.0f} s", flush=True)
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


@app.function(gpu=["L4", "A10"], cpu=8.0, memory=32768, timeout=6 * 3600)
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


@app.local_entrypoint()
def relabel(source: str, destination: str, shard: int = 6) -> None:
  """Regenerate racing labels in CPU shards; require the sampled-clearance audit before publishing."""
  _spawned(relabel_remote.spawn(source, destination, shard))


@app.local_entrypoint()
def racing(source: str = "data/sim/v2", destination: str = "data/sim/racing_v6",
           cfg: str = "configs/train/cuda_racing_v6.yaml") -> None:
  """Run the complete racing-label and training pipeline in Modal."""
  _spawned(racing_remote.spawn(source, destination, cfg))


@app.function(cpu=8.0, memory=16384, timeout=3600)
def transfer_generate_remote(name: str, configurations: int, seconds: float) -> dict:
  import runpy

  _setup()
  try:
    return runpy.run_path(f"{WORK}/scripts/transfer_experiment.py")["generate"](name, configurations, seconds)
  finally:
    vol.commit()


@app.local_entrypoint()
def transfer_generate(name: str = "transfer_pilot_v1", configurations: int = 4, seconds: float = 10.0):
  print(transfer_generate_remote.remote(name, configurations, seconds))


@app.function(cpu=1.0, memory=2048, timeout=7200)
def transfer_pilot_remote() -> dict:
  import json
  import time

  _setup()
  output = Path("runs/transfer_pilot_v1")
  output.mkdir(exist_ok=True)
  start = time.time()
  cache_remote.remote("data/sim/transfer_pilot_v1", "runs/cuda_racing_v6/ckpt/best.pt", "transfer_baseline_v1", False)
  for name, conditioned in (("transfer_pilot_image_v1", False), ("transfer_pilot_conditioned_v1", True)):
    train_remote.remote("configs/train/transfer_pilot_v1.yaml", [f"name={name}", f"conditioned={conditioned}"], True)
  report = {"wall_seconds": time.time() - start, "runs": ["transfer_pilot_image_v1", "transfer_pilot_conditioned_v1"]}
  (output / "timing.json").write_text(json.dumps(report, indent=2))
  vol.commit()
  return report


@app.local_entrypoint()
def transfer_pilot():
  print(transfer_pilot_remote.remote())


@app.function(cpu=2.0, memory=4096, timeout=7200)
def transfer_full_remote(name: str, configurations: int, seconds: float) -> dict:
  import runpy

  _setup()
  api = runpy.run_path(f"{WORK}/scripts/transfer_experiment.py")
  specs = api["prepare_full"](name, configurations, seconds)
  vol.commit()
  # Eight work units, at most three GPU workers; resume by complete episode.
  chunks = [specs[i::8] for i in range(min(8, len(specs)))]
  for rows in transfer_record_shard.map(chunks, kwargs={"out_dir": f"data/sim/{name}"}):
    print(f"completed {len(rows)} episodes", flush=True)
  vol.reload()
  result = api["finalize_full"](name)
  vol.commit()
  return result


@app.local_entrypoint()
def transfer_full(name: str = "transfer_train_v1", configurations: int = 16, seconds: float = 20.0):
  print(transfer_full_remote.remote(name, configurations, seconds))


@app.function(gpu=["L4", "A10"], cpu=4.0, memory=16384, timeout=7200)
def transfer_finish_remote(name: str) -> dict:
  """Resume a stopped generation job with its archived source and exact remaining specs.

  GPU rasterization avoids pathological Mesa software-rendering time at low camera heights.
  Do not run concurrently with generation of the same dataset.
  """
  import json
  import tarfile
  import tempfile
  import time

  _setup()
  root = Path("data/sim") / name
  if (root / "index.json").exists():
    return {"status": "already complete", "index": str(root / "index.json")}
  manifest = json.loads((root / "manifest.json").read_text())
  pending = [s for s in manifest["specs"] if not (root / s["episode_id"] / "meta.json").exists()]
  started = time.time()
  with tempfile.TemporaryDirectory() as directory:
    snapshot = Path(directory)
    with tarfile.open(root / "source.tar.gz") as archive:
      archive.extractall(snapshot, filter="data")
    for folder in LINKED:
      (snapshot / folder).symlink_to(Path(VOL_DIR) / folder)
    (snapshot / "pending.json").write_text(json.dumps(pending))
    code = ("import json,runpy; from fsop.sim.batch_record import EpisodeSpec,_worker; "
            f"_worker(([EpisodeSpec(**s) for s in json.load(open('pending.json'))], 'data/sim/{name}')); "
            f"print(runpy.run_path('scripts/transfer_experiment.py')['finalize_full']('{name}'))")
    try:
      subprocess.run(["python", "-c", code], cwd=snapshot,
                     env={**os.environ, "PYTHONPATH": str(snapshot / "src")}, check=True)
    finally:
      (root / "gpu_resume.json").write_text(json.dumps({"episode_ids": [s["episode_id"] for s in pending],
                                                       "seconds": time.time() - started,
                                                       "source": "source.tar.gz", "renderer": "GPU EGL"}, indent=2))
      vol.commit()
  return {"resumed_episodes": len(pending), "seconds": time.time() - started}


@app.local_entrypoint()
def transfer_finish(name: str = "transfer_train_v1"):
  print(transfer_finish_remote.remote(name))


@app.function(gpu=["L4", "A10"], cpu=4.0, memory=8192, timeout=1800)
def transfer_verify_remote() -> dict:
  import json

  import cv2
  import numpy as np

  _setup()
  from fsop.geometry.camera import CameraModel
  from fsop.profile import load_profile
  from fsop.sim.env import FSEnv, SimCfg
  from fsop.sim.render.panda3d_backend import Panda3DRenderer
  from fsop.track.fixed_layouts import acceleration

  out = Path("runs/transfer_geometry_v1")
  out.mkdir(exist_ok=True)
  subprocess.run(["python", "-c", "import torch, pytest; raise SystemExit(pytest.main(["
                  "'tests/test_reprojection.py', 'tests/test_transfer_geometry.py', 'tests/test_warp.py', '-q']))"],
                 check=True)
  panels, rows = [], []
  for name, z, x, y, pitch in (("rover", .22, -.25, .12, 6), ("rover", .25, .35, -.12, -3),
                               ("formula", 1.65, -.8, .3, 8), ("formula", .75, 1.1, -.3, -3)):
    profile = load_profile(f"configs/profiles/generic_{name}.yaml")
    profile.mount.z, profile.mount.x, profile.mount.y = z, x, y
    profile.mount.pitch = np.deg2rad(pitch)
    cams = {c.name: CameraModel(c.name, c.K, c.width, c.height) for c in profile.cameras}
    renderer = Panda3DRenderer(cams)
    cfg = SimCfg(physical_camera=True, dr="off", randomize_expert=False)
    env = FSEnv(acceleration(width=4), cfg, renderer=renderer, profile=profile)
    env.state.x = 10.0
    obs = env.observe()
    panel = np.hstack([obs.images[c] for c in ("road", "wide")])
    cv2.putText(panel, f"{name}: x={x} y={y} z={z} pitch={pitch}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    panels.append(panel)
    rows.append(profile.to_dict())
    for node in renderer.cam_nps.values():
      node.removeNode()
    for buffer in renderer.buffers.values():
      renderer.base.graphicsEngine.removeWindow(buffer)
    renderer.scene.removeNode()
  cv2.imwrite(str(out / "extremes.jpg"), np.vstack(panels)[..., ::-1])
  (out / "profiles.json").write_text(json.dumps(rows, indent=2))
  vol.commit()
  return {"preview": str(out / "extremes.jpg"), "tests": "passed"}


@app.local_entrypoint()
def transfer_verify():
  print(transfer_verify_remote.remote())


@app.function(gpu=["L4", "A10"], cpu=8.0, memory=32768, timeout=3600)
def transfer_analyze_remote(dataset: str, image_run: str, conditioned_run: str, name: str) -> dict:
  import hashlib
  import json
  import time

  _setup()
  from fsop.data.sim_dataset import Episode, split_episodes
  from fsop.eval.transfer import context_ablation, paired_open_loop, temporal_parity
  from fsop.model.export import export_onnx

  out = Path("runs") / name
  checkpoints = ["runs/cuda_racing_v6/ckpt/best.pt", f"runs/{image_run}/ckpt/best.pt",
                 f"runs/{conditioned_run}/ckpt/best.pt"]
  identity = {"dataset": dataset,
              "source_sha256": _source_identity(),
              "manifest": hashlib.sha256((Path(dataset) / "manifest.json").read_bytes()).hexdigest(),
              "checkpoints": {c: hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in checkpoints}}
  if (out / "identity.json").exists():
    if json.loads((out / "identity.json").read_text()) != identity:
      raise ValueError("Analysis inputs changed; use a distinct run name")
    if (out / "complete.json").exists():
      return json.loads((out / "complete.json").read_text())
  elif out.exists():
    raise ValueError("Existing analysis has no identity; use a distinct run name")
  out.mkdir(exist_ok=True)
  (out / "identity.json").write_text(json.dumps(identity, indent=2))
  _archive_source(out)
  started = time.time()
  try:
    paired_open_loop(dataset, checkpoints, out / "paired_validation.json", split="val")
    context_ablation(dataset, checkpoints[-1], out / "context_ablation.json")
    exported = export_onnx(checkpoints[-1], out / "conditioned.onnx")
    ep = Episode(Path(dataset) / split_episodes(dataset, "val")[0])
    parity = temporal_parity(checkpoints[-1], exported, ep)
    report = {"seconds": time.time() - started, "parity": parity, "export": str(exported)}
    (out / "complete.json").write_text(json.dumps(report, indent=2))
    return report
  finally:
    vol.commit()


@app.local_entrypoint()
def transfer_analyze(dataset: str = "data/sim/transfer_pilot_v1", image_run: str = "transfer_pilot_image_v1",
                     conditioned_run: str = "transfer_pilot_conditioned_v1", name: str = "transfer_pilot_analysis_v1"):
  print(transfer_analyze_remote.remote(dataset, image_run, conditioned_run, name))


@app.function(gpu=["L4", "A10"], cpu=8.0, memory=32768, timeout=3600)
def transfer_open_remote(dataset: str, image_run: str, conditioned_run: str, name: str, extra_run: str) -> dict:
  import hashlib
  import json
  import time

  _setup()
  from fsop.eval.transfer import paired_open_loop

  checkpoints = ["runs/cuda_racing_v6/ckpt/best.pt", f"runs/{image_run}/ckpt/best.pt",
                 f"runs/{conditioned_run}/ckpt/best.pt"]
  if extra_run:
    checkpoints.append(f"runs/{extra_run}/ckpt/best.pt")
  out = Path("runs") / name
  identity = {"source_sha256": _source_identity(), "dataset": dataset,
              "manifest": hashlib.sha256((Path(dataset) / "manifest.json").read_bytes()).hexdigest(),
              "checkpoints": {c: hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in checkpoints}}
  if (out / "identity.json").exists():
    if json.loads((out / "identity.json").read_text()) != identity:
      raise ValueError("Evaluation inputs changed; use a distinct run name")
    if (out / "complete.json").exists():
      return json.loads((out / "complete.json").read_text())
  elif out.exists():
    raise ValueError("Existing evaluation has no identity; use a distinct run name")
  out.mkdir(exist_ok=True)
  (out / "identity.json").write_text(json.dumps(identity, indent=2))
  _archive_source(out)
  started = time.time()
  try:
    result = paired_open_loop(dataset, checkpoints, out / "paired_test.json", split="test")
    report = {"rows": len(result["rows"]), "seconds": time.time() - started}
    (out / "complete.json").write_text(json.dumps(report, indent=2))
    return report
  finally:
    vol.commit()


@app.local_entrypoint()
def transfer_open(dataset: str = "data/sim/transfer_test_v1", image_run: str = "transfer_image_v1",
                  conditioned_run: str = "transfer_conditioned_v1", name: str = "transfer_open_v1",
                  extra_run: str = ""):
  print(transfer_open_remote.remote(dataset, image_run, conditioned_run, name, extra_run))


@app.function(cpu=2.0, memory=4096, timeout=7200)
def transfer_train_remote() -> dict:
  import json
  import runpy
  import time

  _setup()
  started = time.time()
  dataset = "data/sim/transfer_train_v1"
  quality = runpy.run_path(f"{WORK}/scripts/transfer_experiment.py")["audit_dataset"](dataset)
  vol.commit()
  cache_remote.remote(dataset, "runs/cuda_racing_v6/ckpt/best.pt", "transfer_baseline_strict_v1", False)
  for name, conditioned in (("transfer_image_v1", False), ("transfer_conditioned_v1", True)):
    train_remote.remote("configs/train/transfer_v1.yaml", [f"name={name}", f"conditioned={conditioned}"], True)
  result = {"wall_seconds": time.time() - started, "dataset_frames": quality["frames"],
            "runs": ["transfer_image_v1", "transfer_conditioned_v1"]}
  Path("runs/transfer_train_v1").mkdir(exist_ok=True)
  Path("runs/transfer_train_v1/timing.json").write_text(json.dumps(result, indent=2))
  vol.commit()
  return result


@app.local_entrypoint()
def transfer_train():
  print(transfer_train_remote.remote())


@app.function(gpu=["L4", "A10"], cpu=4.0, memory=8192, timeout=1800)
def transfer_audit_remote(dataset: str) -> dict:
  import runpy

  _setup()
  api = runpy.run_path(f"{WORK}/scripts/transfer_experiment.py")
  try:
    quality = api["audit_dataset"](dataset)
    preview = api["preview_dataset"](dataset)
    return {"frames": quality["frames"], "unresolved_fraction": quality["unresolved_fraction"],
            "preview": preview}
  finally:
    vol.commit()


@app.local_entrypoint()
def transfer_audit(dataset: str = "data/sim/transfer_train_v1"):
  print(transfer_audit_remote.remote(dataset))


@app.function(cpu=2.0, memory=4096, timeout=7200)
def transfer_test_data_remote(name: str) -> dict:
  import runpy

  _setup()
  api = runpy.run_path(f"{WORK}/scripts/transfer_experiment.py")
  specs = api["prepare_test"](name)
  vol.commit()
  chunks = [specs[i::6] for i in range(min(6, len(specs)))]
  for rows in transfer_record_shard.map(chunks, kwargs={"out_dir": f"data/sim/{name}"}):
    print(f"completed {len(rows)} final-test episodes", flush=True)
  vol.reload()
  result = api["finalize_test"](name)
  vol.commit()
  return result


@app.local_entrypoint()
def transfer_test_data(name: str = "transfer_test_v1"):
  print(transfer_test_data_remote.remote(name))


@app.function(gpu=["L4", "A10"], cpu=4.0, memory=16384, timeout=3600, max_containers=3)
def transfer_case_remote(configuration: dict, checkpoints: list[str], track_seed: int,
                         scene_seed: int, recovery: bool, name: str, max_seconds: float = 100.) -> list[dict]:
  import hashlib
  import json

  _setup()
  vol.reload()  # reused workers must see completed cases after orchestrator retries
  from fsop.eval.transfer import closed_loop_case

  root = Path("runs") / name
  root.mkdir(exist_ok=True)
  path = root / f"{configuration['name']}_{track_seed}_{scene_seed}.json"
  identity = {"configuration": configuration, "track_seed": track_seed, "scene_seed": scene_seed,
              "source_sha256": _source_identity(),
              "recovery": recovery, "max_seconds": max_seconds,
              "checkpoints": {c: hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in checkpoints}}
  if path.exists():
    saved = json.loads(path.read_text())
    if saved["identity"] != identity:
      raise ValueError("Evaluation identity differs; use a distinct run name")
    return saved["results"]
  try:
    results = closed_loop_case(configuration, checkpoints, track_seed, scene_seed, recovery, max_seconds)
    path.write_text(json.dumps({"identity": identity, "results": results}, indent=2))
    return results
  finally:
    vol.commit()


@app.function(cpu=1.0, memory=2048, timeout=10800)
def transfer_closed_remote(image_run: str, conditioned_run: str, name: str, runtime: bool,
                           export_run: str, extra_run: str) -> dict:
  import json

  import yaml

  _setup()
  cfg = yaml.safe_load(Path("configs/eval/transfer_v1.yaml").read_text())
  checkpoints = ["runs/cuda_racing_v6/ckpt/best.pt", f"runs/{image_run}/ckpt/best.pt",
                 f"runs/{conditioned_run}/ckpt/best.pt"]
  if extra_run:
    checkpoints.append(f"runs/{extra_run}/ckpt/best.pt")
  configurations = cfg["configurations"]
  if runtime:
    checkpoints = [f"runs/{conditioned_run}/ckpt/best.pt", f"runs/{export_run}/conditioned.onnx"]
    configurations = [c for c in configurations if c["name"] in ("formula_high_rear", "rover_low_forward")]
  args = [(c, checkpoints, ts, ss, bool(i), name) for c in configurations
          for i, (ts, ss) in enumerate(zip(cfg["track_seeds"], cfg["scene_seeds"], strict=True))]
  results = []
  for rows in transfer_case_remote.starmap(args):
    results.extend(rows)
  vol.reload()
  out = Path("runs") / name
  out.mkdir(exist_ok=True)
  _archive_source(out)
  report = {"suite": cfg, "runtime": runtime, "results": results}
  (out / "report.json").write_text(json.dumps(report, indent=2))
  vol.commit()
  return {"runs": len(results), "report": str(out / "report.json")}


@app.local_entrypoint()
def transfer_closed(image_run: str = "transfer_image_v1", conditioned_run: str = "transfer_conditioned_v1",
                    name: str = "transfer_closed_v1", runtime: bool = False,
                    export_run: str = "transfer_analysis_v1", extra_run: str = ""):
  print(transfer_closed_remote.remote(image_run, conditioned_run, name, runtime, export_run, extra_run))


@app.function(gpu=["L4", "A10"], cpu=4.0, memory=16384, timeout=1800, max_containers=3)
def transfer_v2_collect_remote(spec: dict, root: str) -> dict:
  _setup()
  vol.reload()
  from fsop.sim.transfer_collection import collect
  try:
    return collect(spec, root)
  finally:
    vol.commit()


@app.function(cpu=2.0, memory=4096, timeout=7200)
def transfer_v2_data_remote(name: str, pilot: bool) -> dict:
  import runpy

  _setup()
  api = runpy.run_path(f"{WORK}/scripts/transfer_v2_experiment.py")
  specs = api["prepare"](name, pilot)
  vol.commit()
  rows = list(transfer_v2_collect_remote.starmap([(spec, f"data/sim/{name}") for spec in specs]))
  vol.reload()
  result = api["finish"](name, rows)
  vol.commit()
  return {k: v for k, v in result.items() if k != "rows"}


@app.local_entrypoint()
def transfer_v2_data(name: str = "transfer_v2_pilot", pilot: bool = True):
  print(transfer_v2_data_remote.remote(name, pilot))


@app.function(cpu=2, memory=4096, timeout=7200)
def transfer_v2_pilot_train_remote() -> dict:
  _setup()
  cache_remote.remote('data/sim/transfer_v2_pilot', 'runs/cuda_racing_v6/ckpt/best.pt',
                      'transfer_baseline_strict_v2_full', False)
  train_remote.remote('configs/train/transfer_pilot_v2.yaml', [], True)
  return {'run': 'transfer_pilot_v2_retry1', 'steps': 300}


@app.local_entrypoint()
def transfer_v2_pilot_train():
  print(transfer_v2_pilot_train_remote.remote())


@app.function(cpu=2, memory=4096, timeout=14400)
def transfer_suite_remote(suite: str, checkpoints: list[str], name: str) -> dict:
  import json

  import yaml
  _setup()
  cfg = yaml.safe_load(Path(suite).read_text())
  out = Path('runs') / name
  out.mkdir(exist_ok=True)
  _archive_source(out)  # retain the exact evaluator even if a worker fails later
  vol.commit()
  args = [(c, checkpoints, ts, ss, bool(i) if len(cfg['track_seeds']) > 1 else bool(j % 2), name,
           180. if cfg.get('version') == 2 and c['family'] == 'rover' else 100.)
          for j, c in enumerate(cfg['configurations'])
          for i, (ts, ss) in enumerate(zip(cfg['track_seeds'], cfg['scene_seeds'], strict=True))]
  results = []
  for rows in transfer_case_remote.starmap(args):
    results.extend(rows)
  vol.reload()
  report = {'suite': cfg, 'suite_path': suite, 'checkpoints': checkpoints,
            'time_limits_seconds': {'formula': 100., 'rover': 180. if cfg.get('version') == 2 else 100.},
            'recovery_rule': 'second seed, or odd configuration for a single-seed validation suite',
            'results': results}
  (out / 'report.json').write_text(json.dumps(report, indent=2))
  vol.commit()
  return {'runs': len(results), 'report': str(out / 'report.json')}


@app.local_entrypoint()
def transfer_suite(suite: str, checkpoints: str, name: str):
  print(transfer_suite_remote.remote(suite, checkpoints.split(','), name))


@app.function(cpu=2, memory=4096, timeout=14400)
def transfer_v2_train_remote() -> dict:
  _setup()
  cache_remote.remote('data/sim/transfer_train_v2', 'runs/cuda_racing_v6/ckpt/best.pt',
                      'transfer_baseline_strict_v2_full', False)
  names = ['transfer_expert_v2', 'transfer_recovery_v2', 'transfer_sequence_v2']
  for name in names:
    train_remote.remote(f'configs/train/{name}.yaml', [], True)
  return {'runs': names, 'steps_each': 1500}


@app.local_entrypoint()
def transfer_v2_train():
  print(transfer_v2_train_remote.remote())


@app.function(cpu=2, memory=4096, timeout=7200)
def transfer_v2_final_data_remote(selection: str, name: str) -> dict:
  import runpy
  _setup()
  api = runpy.run_path('scripts/transfer_v2_test_data.py')
  specs = api['prepare'](selection, name)
  vol.commit()
  rows = list(transfer_v2_collect_remote.starmap([(spec, f'data/sim/{name}') for spec in specs]))
  vol.reload()
  report = api['finish'](name, specs, rows)
  vol.commit()
  return report


@app.local_entrypoint()
def transfer_v2_final_data(selection: str, name: str = 'transfer_test_v2'):
  print(transfer_v2_final_data_remote.remote(selection, name))


@app.local_entrypoint()
def transfer_v2_policy_train():
  """The single predeclared fallback: train policy slowly, retain frozen vision."""
  train_remote.remote('configs/train/transfer_policy_v2.yaml', [], True)


@app.local_entrypoint()
def transfer_v3_pilot_train():
  train_remote.remote('configs/train/transfer_control_pilot_v3.yaml', [], True)


@app.local_entrypoint()
def transfer_v3_control_train():
  train_remote.remote('configs/train/transfer_control_v3.yaml', [], True)


@app.function(cpu=4, memory=8192, timeout=7200)
def transfer_v3_reference_data_remote(guarded: bool = False) -> dict:
  import runpy

  _setup()
  api = runpy.run_path('scripts/transfer_v3_recovery_data.py')
  name = 'transfer_guarded_train_v3' if guarded else api['NAME']
  policy = 'guarded' if guarded else 'reference'
  specs = api['prepare'](name, f'configs/distributions/transfer_collection_{policy}_v3.yaml')
  vol.commit()
  rows = list(transfer_v2_collect_remote.starmap(
      [(spec, f"data/sim/{name}") for spec in specs]))
  vol.reload()
  report = api['finish'](rows, name)
  vol.commit()
  return {k:v for k,v in report.items() if k != 'rows'}


@app.local_entrypoint()
def transfer_v3_reference_data(guarded: bool = False):
  print(transfer_v3_reference_data_remote.remote(guarded))


@app.local_entrypoint()
def transfer_v3_reference_pilot():
  cache_remote.remote('data/sim/transfer_reference_train_v3', 'runs/cuda_racing_v6/ckpt/best.pt',
                      'transfer_baseline_strict_v2_full', False)
  train_remote.remote('configs/train/transfer_reference_pilot_v3.yaml', [], True)


@app.local_entrypoint()
def transfer_v3_reference_train():
  train_remote.remote('configs/train/transfer_reference_v3.yaml', [], True)


@app.local_entrypoint()
def transfer_v3_guarded_pilot():
  cache_remote.remote('data/sim/transfer_guarded_train_v3', 'runs/cuda_racing_v6/ckpt/best.pt',
                      'transfer_baseline_strict_v2_full', False)
  train_remote.remote('configs/train/transfer_guarded_pilot_v3.yaml', [], True)


@app.local_entrypoint()
def transfer_v3_guarded_train():
  train_remote.remote('configs/train/transfer_guarded_v3.yaml', [], True)


@app.local_entrypoint()
def transfer_v4_vision_train(pilot: bool = True):
  suffix = 'pilot_v4' if pilot else 'v4'
  calls = [train_remote.spawn(f'configs/train/transfer_vision_{mode}_{suffix}.yaml', [], True)
           for mode in ('frozen', 'last')]
  failures = []
  for call in calls:
    try:
      call.get()
    except Exception as exc:
      failures.append(str(exc))
  if failures:
    raise RuntimeError(f'Vision comparison failures: {failures}')


@app.local_entrypoint()
def transfer_v5_state_pilots():
  # Preserve the old cache and its provenance. The diagnostic validates identical
  # vision tensors before this separate call stages either training job.
  bench_remote.remote('python scripts/transfer_v5_cache_check.py')
  calls = [train_remote.spawn(f'configs/train/transfer_state_{mode}_pilot_v5.yaml', [], True)
           for mode in ('constant', 'speed')]
  failures = []
  for call in calls:
    try:
      call.get()
    except Exception as exc:
      failures.append(str(exc))
  if failures:
    raise RuntimeError(f'State comparison failures: {failures}')


@app.local_entrypoint()
def transfer_v5_state_train():
  calls = [train_remote.spawn(f'configs/train/transfer_state_{mode}_v5.yaml', [], True)
           for mode in ('constant', 'speed')]
  failures = []
  for call in calls:
    try:
      call.get()
    except Exception as exc:
      failures.append(str(exc))
  if failures:
    raise RuntimeError(f'State full comparison failures: {failures}')


@app.function(cpu=2.0, memory=4096, timeout=600)
def transfer_v5_coverage_remote():
  _setup()
  try:
    _sh('python scripts/transfer_v5_coverage_audit.py')
  finally:
    vol.commit()


@app.local_entrypoint()
def transfer_v5_coverage():
  transfer_v5_coverage_remote.remote()


@app.function(cpu=2.0, memory=4096, timeout=600)
def transfer_feature_audit_remote():
  _setup()
  try:
    _sh('python scripts/transfer_feature_norm_audit.py')
  finally:
    vol.commit()


@app.local_entrypoint()
def transfer_feature_audit():
  transfer_feature_audit_remote.remote()


@app.local_entrypoint()
def transfer_norm_interventions():
  pairs = [('runs/transfer_minimal_v1/ckpt/best.pt', 'runs/transfer_norm_v1_probe_v6'),
           ('runs/transfer_state_speed_v5/ckpt/step_000250.pt', 'runs/transfer_norm_speed_probe_v6')]
  calls = [bench_remote.spawn(f'python scripts/transfer_norm_intervention.py {checkpoint} {out}')
           for checkpoint, out in pairs]
  failures = []
  for call in calls:
    try:
      call.get()
    except Exception as exc:
      failures.append(str(exc))
  if failures:
    raise RuntimeError(f'Norm intervention failures: {failures}')


@app.local_entrypoint()
def transfer_gradient_audit():
  bench_remote.remote('python scripts/transfer_gradient_audit.py')


@app.function(cpu=2, memory=4096, timeout=7200)
def transfer_v7_refresh_remote():
  import runpy

  _setup()
  api = runpy.run_path('scripts/transfer_v7_refresh.py')
  specs = api['prepare']()
  vol.commit()
  rows = list(transfer_v2_collect_remote.starmap([(s, str(api['ROOT'])) for s in specs]))
  vol.reload()
  report = api['finish'](rows)
  vol.commit()
  return report


@app.local_entrypoint()
def transfer_v7_refresh():
  print(transfer_v7_refresh_remote.remote())


@app.function(cpu=2, memory=4096, timeout=600)
def transfer_v7_inspect_remote():
  _setup()
  try:
    _sh('python scripts/transfer_v7_inspect.py')
  finally:
    vol.commit()


@app.local_entrypoint()
def transfer_v7_inspect():
  transfer_v7_inspect_remote.remote()


@app.function(cpu=4, memory=8192, timeout=1200)
def transfer_v7_expert_control_remote():
  _setup()
  try:
    _sh('python scripts/transfer_v7_expert_control.py')
  finally:
    vol.commit()


@app.local_entrypoint()
def transfer_v7_expert_control():
  transfer_v7_expert_control_remote.remote()


@app.local_entrypoint()
def transfer_v7_guard_probe():
  bench_remote.remote('python scripts/transfer_v7_guard_probe.py')


@app.function(cpu=2, memory=4096, timeout=7200)
def transfer_v7_guarded_refresh_remote():
  import json
  import runpy

  _setup()
  api = runpy.run_path('scripts/transfer_v7_guarded_refresh.py')
  specs = api['prepare']()
  vol.commit()
  rows = list(transfer_v2_collect_remote.starmap([(s, str(api['ROOT'])) for s in specs]))
  vol.reload()
  retained = json.loads((api['ROOT']/'retained_rows.json').read_text())
  report = runpy.run_path('scripts/transfer_v7_refresh.py')['finish'](retained+rows, api['ROOT'])
  _sh(f"python scripts/transfer_v7_inspect.py {api['ROOT']}")
  vol.commit()
  return report


@app.local_entrypoint()
def transfer_v7_guarded_refresh():
  print(transfer_v7_guarded_refresh_remote.remote())


@app.function(cpu=4, memory=8192, timeout=3600)
def transfer_v7_stage_remote():
  _setup()
  try:
    _sh('python scripts/transfer_v7_stage.py')
  finally:
    vol.commit()


@app.local_entrypoint()
def transfer_v7_stage():
  transfer_v7_stage_remote.remote()


@app.local_entrypoint()
def transfer_v7_train():
  for arm in ('previous', 'current'):
    cache_remote.remote(f'data/sim/transfer_refresh_{arm}_train_v7',
                        'runs/cuda_racing_v6/ckpt/best.pt', 'transfer_baseline_strict_v2_full', False)
  calls = [train_remote.spawn(f'configs/train/transfer_refresh_{arm}_pilot_v7.yaml', [], True)
           for arm in ('previous', 'current')]
  failures = []
  for call in calls:
    try:
      call.get()
    except Exception as exc:
      failures.append(str(exc))
  if failures:
    raise RuntimeError(f'Refresh paired pilot failures: {failures}')


@app.local_entrypoint()
def transfer_v7_control_retry():
  train_remote.remote('configs/train/transfer_refresh_previous_pilot_v7_retry1.yaml', [], True)
