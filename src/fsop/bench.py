"""Benchmarks: versioned result records + generated tables.

Each benchmark is one JSON file in ``benchmarks/results/`` (committed) containing the open-loop summary
on the held-out test tracks, the closed-loop suite aggregate and provenance (hardware, git SHA, data,
training config). ``fsop bench table`` regenerates ``benchmarks/README.md`` (full history) and the
summary table in the top-level README between the ``BENCHMARKS`` markers.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

RESULTS = Path("benchmarks/results")
START, END = "<!-- BENCHMARKS:START -->", "<!-- BENCHMARKS:END -->"
OL_KEYS = ("lat_0.5s", "lat_1.0s", "lat_2.0s", "long_1.0s", "long_2.0s", "ade_4s", "heading_1s_deg",
           "bev_f1", "bev_imgproj_f1", "bev_imgproj_pos_err", "n_frames")


def hardware() -> dict:
  info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
  try:
    if platform.system() == "Darwin":
      info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
      info["memory_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30)
  except Exception:  # noqa: BLE001
    pass
  try:
    import torch

    info["torch"] = torch.__version__
    if torch.cuda.is_available():
      info["gpu"] = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
      info["gpu"] = "Apple MPS"
  except ImportError:
    pass
  return info


def git_sha() -> str:
  try:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
    return sha + ("-dirty" if dirty else "")
  except Exception:  # noqa: BLE001
    return "unknown"


def _train_info(ckpt: str | None) -> dict:
  if not ckpt or not Path(ckpt).exists():
    return {}
  import torch

  sd = torch.load(ckpt, map_location="cpu", weights_only=False)
  cfg = sd.get("cfg", {})
  return {"checkpoint": str(ckpt), "step": sd.get("step"), "stage": cfg.get("stage"), "trunk": cfg.get("trunk"),
          "batch_size": cfg.get("batch_size"), "init_ckpt": cfg.get("init_ckpt"), "sim_root": cfg.get("sim_root"),
          "fsoco": bool(cfg.get("fsoco_root"))}


def _data_info(dataset: str | None) -> dict:
  if not dataset or not (Path(dataset) / "index.json").exists():
    return {}
  eps = json.loads((Path(dataset) / "index.json").read_text())["episodes"]
  return {"dataset": str(dataset), "episodes": len(eps), "frames_5hz": sum(e["frames"] for e in eps),
          "tracks": len({e["track_id"] for e in eps})}


def make_record(name: str, model: str, open_loop: dict | None, closed_loop: dict | None, ckpt: str | None = None,
                dataset: str | None = None, notes: str = "", train_hardware: str = "", date: str | None = None) -> dict:
  ol = {}
  if open_loop:
    s = open_loop["summary"]["all"]
    ol = {k: s.get(k) for k in OL_KEYS if s.get(k) is not None}
  cl = {}
  if closed_loop:
    cl = {"suite": closed_loop.get("suite", {}), "profile": closed_loop.get("profile", "default"),
          "aggregate": closed_loop["aggregate"]}
  return {
    "name": name, "model": model, "date": date or dt.date.today().isoformat(), "git": git_sha(),
    "eval_hardware": hardware(), "train_hardware": train_hardware, "training": _train_info(ckpt),
    "data": _data_info(dataset), "open_loop": ol, "closed_loop": cl, "notes": notes, "schema": 1,
  }


def save_record(rec: dict) -> Path:
  RESULTS.mkdir(parents=True, exist_ok=True)
  p = RESULTS / f"{rec['date']}_{rec['name']}.json"
  p.write_text(json.dumps(rec, indent=1, default=float) + "\n")
  return p


def load_records() -> list[dict]:
  return sorted((json.loads(p.read_text()) for p in RESULTS.glob("*.json")), key=lambda r: (r["date"], r["name"]))


def _f(v, fmt="{:.2f}"):
  return "–" if v is None else fmt.format(v)


def _cl(rec: dict, disc: str) -> str:
  agg = rec.get("closed_loop", {}).get("aggregate", {})
  rows = [v for k, v in agg.items() if k.startswith(disc + "/")]
  if not rows:
    return "–"
  passed = sum(r["pass_rate"] * r["runs"] for r in rows)
  runs = sum(r["runs"] for r in rows)
  s = f"{passed:.0f}/{runs}"
  if disc == "trackdrive":
    doo = sum(r["doo_per_run"] * r["runs"] for r in rows) / runs
    s += f" ({doo:.1f} DOO)"
  return s


def summary_table(records: list[dict]) -> str:
  head = ("| date | name | model | train hw | lat err 1 s / 2 s [m] | ADE 4 s [m] | heading 1 s [°] | cone F1 | "
          "trackdrive | accel | skidpad |\n|---|---|---|---|---|---|---|---|---|---|---|")
  lines = [head]
  for r in records:
    o = r.get("open_loop", {})
    f1 = o.get("bev_imgproj_f1", o.get("bev_f1"))
    lines.append(f"| {r['date']} | {r['name']} | {r['model']} | {r.get('train_hardware') or '–'} | "
                 f"{_f(o.get('lat_1.0s'))} / {_f(o.get('lat_2.0s'))} | {_f(o.get('ade_4s'))} | "
                 f"{_f(o.get('heading_1s_deg'), '{:.1f}')} | {_f(f1)} | {_cl(r, 'trackdrive')} | "
                 f"{_cl(r, 'acceleration')} | {_cl(r, 'skidpad')} |")
  return "\n".join(lines)


def write_tables(readme: Path = Path("README.md")) -> None:
  recs = load_records()
  table = summary_table(recs)
  bench_md = ["# Benchmarks", "",
              "Generated by `fsop bench table` from `benchmarks/results/*.json` - do not edit by hand.", "",
              "**Protocol** (see [docs/benchmarks.md](../docs/benchmarks.md)): open-loop metrics on the held-out",
              "*test* tracks of the sim dataset, closed-loop quick suite (`configs/eval/closed_loop_quick.yaml`).", "",
              table, ""]
  for r in recs:
    bench_md += [f"## {r['date']} - {r['name']}", "", f"- model: {r['model']}  ", f"- git: `{r['git']}`  ",
                 f"- training: `{json.dumps(r.get('training', {}))}`  ", f"- data: `{json.dumps(r.get('data', {}))}`  ",
                 f"- eval hardware: `{json.dumps(r.get('eval_hardware', {}))}`  "]
    if r.get("notes"):
      bench_md += [f"- notes: {r['notes']}"]
    bench_md += [""]
  Path("benchmarks").mkdir(exist_ok=True)
  Path("benchmarks/README.md").write_text("\n".join(bench_md))
  text = readme.read_text()
  if START in text and END in text:
    a, b = text.index(START) + len(START), text.index(END)
    readme.write_text(text[:a] + "\n" + table + "\n" + text[b:])
