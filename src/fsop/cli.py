"""fsop command line: fsop <group> <command> --help"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Formula Student openpilot pipeline")
track_app = typer.Typer(no_args_is_help=True, help="Track generation / validation / export")
sim_app = typer.Typer(no_args_is_help=True, help="Simulation: live driving and dataset recording")
model_app = typer.Typer(no_args_is_help=True, help="Fetch / convert / verify the openpilot model")
data_app = typer.Typer(no_args_is_help=True, help="Datasets: FSOCO import, feature caches")
eval_app = typer.Typer(no_args_is_help=True, help="Open- and closed-loop evaluation")
viz_app = typer.Typer(no_args_is_help=True, help="Visualization helpers")
for name, sub in (("track", track_app), ("sim", sim_app), ("model", model_app), ("data", data_app),
                  ("eval", eval_app), ("viz", viz_app)):
  app.add_typer(sub, name=name)

DEFAULT_ONNX = "artifacts/models/driving_supercombo_b361e952.onnx"
PREP_DIR = "artifacts/models/prep"


# ---------------------------------------------------------------------------------------------- track
@track_app.command("gen")
def track_gen(out: Path = typer.Option(..., help="output directory"), n: int = 100, seed: int = 0,
              cfg: Path | None = typer.Option(None, help="TrackGenCfg yaml"), workers: int = 6,
              override: list[str] = typer.Option([], "-o", help="dot-list overrides, e.g. -o length_max=400")):
  """Generate a set of rule-compliant trackdrive/autocross tracks."""
  from fsop.config import load_config
  from fsop.track.batch import generate_set
  from fsop.track.generate import TrackGenCfg

  c = load_config(TrackGenCfg, cfg, override)
  rows = generate_set(out, n, c, seed0=seed, workers=workers)
  typer.echo(f"wrote {len(rows)} tracks to {out}")


@track_app.command("fixed")
def track_fixed(out: Path = Path("data/tracks/fixed")):
  """Write the fixed skidpad and acceleration layouts."""
  from fsop.track.fixed_layouts import acceleration, skidpad
  from fsop.track.io import save_json

  for t in (skidpad(), acceleration()):
    save_json(t, out / f"{t.id}.json")
  typer.echo(f"wrote skidpad + acceleration to {out}")


@track_app.command("validate")
def track_validate(path: Path):
  """Validate a track json or a generated set."""
  from fsop.track.batch import load_set
  from fsop.track.validate import validate

  bad = 0
  for t in load_set(path):
    r = validate(t)
    if not r.passed:
      bad += 1
      typer.echo(str(r))
  typer.echo(f"{bad} failing tracks")
  raise typer.Exit(1 if bad else 0)


@track_app.command("export")
def track_export(track: Path, out: Path, fmt: str = typer.Option("eufs", help="eufs | fsds")):
  from fsop.track.io import load_json, to_cone_csv

  to_cone_csv(load_json(track), out, fmt)


@track_app.command("preview")
def track_preview(path: Path, png: Path | None = None, rerun: bool = False, limit: int = 16):
  """PNG gallery (with validator status) and/or Rerun view of tracks."""
  from fsop.track.batch import load_set

  tracks = load_set(path)[:limit]
  if png:
    from fsop.viz.track_preview import gallery

    typer.echo(f"wrote {gallery(tracks, png)}")
  if rerun:
    import rerun as rr

    from fsop.viz import rr_log

    rr_log.init("fsop_tracks", spawn=True)
    for i, t in enumerate(tracks):
      rr.set_time("track", sequence=i)
      rr_log.log_track(t, path=f"tracks/{t.id}")


# ---------------------------------------------------------------------------------------------- sim
@sim_app.command("drive")
def sim_drive(track: str = typer.Option("skidpad", help="track json | skidpad | acceleration"),
              dr: str = "light", seed: int = 0, noise: bool = False, ckpt: Path | None = None,
              save: Path | None = typer.Option(None, help="write .rrd instead of spawning the viewer"),
              max_seconds: float = 0.0, controller: str = "pp"):
  """Drive one episode (expert, or a checkpoint with --ckpt) with Rerun visualization."""
  from fsop.sim.drive import expert_policy, run_episode
  from fsop.sim.env import FSEnv, SimCfg
  from fsop.viz import rr_log

  t = _load_track(track)
  cfg = SimCfg(dr=dr, max_time=max_seconds)
  cfg.noise.enabled = noise
  env = FSEnv(t, cfg, seed=seed)
  policy = expert_policy
  if ckpt is not None:
    from fsop.eval.closed_loop import ModelPolicy

    policy = ModelPolicy.from_checkpoint(ckpt, controller=controller)
  rr_log.init("fsop_sim", spawn=save is None, save=str(save) if save else None,
              blueprint=rr_log.sim_blueprint(bev=ckpt is not None))
  summary = run_episode(env, policy, viz=True, add_noise=noise)
  typer.echo(json.dumps(summary, indent=1, default=float))


@sim_app.command("record")
def sim_record(tracks: Path = typer.Option(..., help="track set directory (fsop track gen)"),
               out: Path = typer.Option(...), workers: int = 4, seconds: float = 40.0, seed: int = 0,
               skidpad: int = 10, accel: int = 10, standing: float = 0.15):
  """Record an expert dataset (noise injection, DR mix, counterfactual plan labels)."""
  from fsop.sim.batch_record import plan_dataset, record_dataset

  specs = plan_dataset(str(tracks), n_standing=standing, n_skidpad=skidpad, n_accel=accel, max_seconds=seconds,
                       seed=seed)
  idx = record_dataset(specs, str(out), workers=workers)
  typer.echo(f"{len(idx)} episodes, {sum(e['frames'] for e in idx)} frames in {out}")


def _load_track(track: str):
  from fsop.track.fixed_layouts import acceleration, skidpad
  from fsop.track.generate import generate_track
  from fsop.track.io import load_json

  if track == "skidpad":
    return skidpad()
  if track == "acceleration":
    return acceleration()
  if track.startswith("gen:"):
    return generate_track(int(track.split(":")[1]))
  return load_json(track)


# ---------------------------------------------------------------------------------------------- model
@model_app.command("fetch")
def model_fetch(out: Path = Path(DEFAULT_ONNX)):
  """Download openpilot's driving_supercombo.onnx (pinned commit) and check its sha256."""
  from fsop.model.fetch import fetch

  typer.echo(fetch(out))


@model_app.command("prepare")
def model_prepare(src: Path = Path(DEFAULT_ONNX), out: Path = Path(PREP_DIR)):
  """FP32 cast, batchable reshapes, taps, vision/policy split."""
  from fsop.model.onnx_prep import prepare

  typer.echo(prepare(src, out))


@model_app.command("verify")
def model_verify(src: Path = Path(DEFAULT_ONNX), prep: Path = Path(PREP_DIR),
                 out: Path = Path("artifacts/models/verify_report.json")):
  """Numerical parity: ORT fp16 / fp32 vs converted torch on CPU and MPS."""
  from fsop.model.verify import verify

  r = verify(src, prep, out=out)
  typer.echo(json.dumps({k: v["plan"] for k, v in r.items()}, indent=1))


# ---------------------------------------------------------------------------------------------- data
@data_app.command("fsoco-import")
def data_fsoco(src: Path = typer.Option(..., help="FSOCO root (Supervisely export or YOLO export)"),
               out: Path = Path("data/fsoco"), fmt: str = "auto"):
  """Index FSOCO into data/fsoco/index.json with a team-level split."""
  from fsop.data.fsoco import import_fsoco

  idx = import_fsoco(src, out, fmt)
  typer.echo(f"{len(idx['images'])} images, splits: { {k: len(v) for k, v in idx['splits'].items()} }")


@data_app.command("cache-features")
def data_cache(dataset: Path = typer.Option(...), ckpt: Path | None = None, batch: int = 32):
  """Run the (frozen) vision trunk once per frame and cache hidden states + spatial taps."""
  from fsop.data.feature_cache import build_cache

  typer.echo(build_cache(dataset, ckpt, batch=batch))


# ---------------------------------------------------------------------------------------------- train / eval
@app.command("train")
def train(cfg: Path = typer.Option(...), override: list[str] = typer.Option([], "-o")):
  """Train (stage set in the config)."""
  from fsop.config import load_config
  from fsop.train.trainer import TrainCfg, Trainer

  c = load_config(TrainCfg, cfg, override)
  Trainer(c).fit()


@eval_app.command("open-loop")
def eval_open_loop(dataset: Path = typer.Option(...), ckpt: Path | None = None, split: str = "test",
                   out: Path | None = None, rrd: Path | None = None, max_episodes: int = 0):
  """Open-loop plan + BEV metrics on held-out sim tracks (ckpt omitted = untouched openpilot model)."""
  from fsop.eval.open_loop import evaluate

  r = evaluate(dataset, ckpt, split=split, rrd=rrd, max_episodes=max_episodes)
  if out:
    out.write_text(json.dumps(r, indent=1, default=float))
  typer.echo(json.dumps(r["summary"], indent=1, default=float))


@eval_app.command("closed-loop")
def eval_closed_loop(ckpt: Path | None = None, suite: Path = Path("configs/eval/closed_loop_suite.yaml"),
                     out: Path = Path("runs/closed_loop"), controller: str = "pp", rrd: bool = True,
                     baseline: str = typer.Option(
                       "model", help="model | openpilot | geometric_gt | geometric_model | expert")):
  """Closed-loop suite in the simulator with the FSG referee."""
  from fsop.eval.closed_loop import run_suite

  r = run_suite(suite, ckpt, out, controller=controller, save_rrd=rrd, baseline=baseline)
  typer.echo(json.dumps(r["aggregate"], indent=1, default=float))


@eval_app.command("dagger")
def eval_dagger(ckpt: Path = typer.Option(...), tracks: Path = Path("data/tracks/td_v1"),
                dataset: Path = Path("data/sim/v1"), n: int = 20, seconds: float = 40.0, beta: float = 0.2,
                dr: str = "light", seed: int = 0):
  """DAgger round: the checkpoint drives train-split tracks, the expert labels; appends to the dataset."""
  from fsop.eval.dagger import append_to_index, dagger_round
  from fsop.track.batch import split_ids
  from fsop.track.io import load_json

  idx = json.loads((tracks / "index.json").read_text())["tracks"]
  train = set(split_ids(idx)["train"])
  sel = [load_json(tracks / r["path"]) for r in idx if r["id"] in train][:n]
  rows = dagger_round(ckpt, sel, dataset, seconds=seconds, beta=beta, dr=dr, seed=seed)
  append_to_index(dataset, rows)
  typer.echo(f"added {len(rows)} DAgger episodes ({sum(r['frames'] for r in rows)} frames) to {dataset}; "
             "re-run `fsop data cache-features` before stage-1 training")


@app.command("export")
def export(ckpt: Path = typer.Option(...), out: Path = Path("artifacts/export/fs_supercombo.onnx")):
  """Export the fine-tuned model to ONNX (original 2576 outputs + appended cone outputs)."""
  from fsop.model.export import export_onnx

  typer.echo(export_onnx(ckpt, out))


@viz_app.command("dataset")
def viz_dataset(episode: Path, save: Path | None = None, ckpt: Path | None = None, every: int = 1):
  """Replay a recorded episode in Rerun: frames, GT plan, GT cones / BEV targets (+ predictions)."""
  from fsop.viz.inspect import inspect_episode

  inspect_episode(episode, save=save, ckpt=ckpt, every=every)


@viz_app.command("video")
def viz_video(ckpt: Path = typer.Option(...),
              track: str = typer.Option("gen:7", help="track json | skidpad | acceleration | gen:<seed>"),
              out: Path = Path("artifacts/videos/run.mp4"), laps: int = 1, seconds: float = 90.0, dr: str = "light",
              seed: int = 0):
  """Closed-loop MP4: cameras with projected plan + detected cones, track map, ego map, HUD."""
  from fsop.viz.video import render_run

  t = _load_track(track)
  if t.discipline == "trackdrive":
    t.laps = laps
  r = render_run(ckpt, t, out, seconds=seconds, dr=dr, seed=seed)
  typer.echo(json.dumps({k: r[k] for k in ("path", "frames", "result", "doo", "oc", "laps")}, default=float))


if __name__ == "__main__":
  app()
