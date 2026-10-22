"""Closed-loop evaluation in the simulator with the FSG referee.

Loop at 20 Hz: render -> YUV pack -> TemporalState (openpilot queues) -> model -> plan follower ->
dynamics -> referee. Policies (``baseline``): ``model`` (fine-tuned checkpoint), ``openpilot``
(untouched model), ``geometric_gt`` (cone midline planner on visible GT cones = perfect perception),
``geometric_model`` (midline planner on the checkpoint's BEV detections), ``expert`` (privileged).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from fsop.eval.controller import make_controller
from fsop.geometry.op_frames import op_to_ego_plan
from fsop.geometry.yuv import rgb_to_model
from fsop.model.temporal import TemporalState
from fsop.sim.env import FSEnv, Obs, SimCfg
from fsop.train.device import pick_device


class ModelPolicy:
  """Runs SupercomboFS (or the untouched base model) at 20 Hz with openpilot's queue semantics."""

  def __init__(self, model, controller: str = "pp", device=None, bev: bool = True, cones: str = "img"):
    self.cones = cones  # img: image head + ground projection | bev: learned BEV head
    self.device = device or pick_device()
    self.model = model.to(self.device).eval()
    self.controller_name = controller
    self.ctrl = None
    self.ts = TemporalState()
    self.bev = bev and hasattr(model, "bev")

  @classmethod
  def from_checkpoint(cls, ckpt: str | Path | None, controller: str = "pp") -> ModelPolicy:
    if ckpt is None:
      from fsop.model.build import build_base

      return cls(build_base(None), controller, bev=False)
    from fsop.model.build import load_fs

    return cls(load_fs(ckpt), controller)

  def reset(self) -> None:
    self.ts = TemporalState()
    self.ctrl = None

  @torch.no_grad()
  def __call__(self, obs: Obs, env: FSEnv) -> tuple[float, float, dict]:
    if self.ctrl is None:
      self.ctrl = make_controller(self.controller_name, env.vp)
    frames = {c: rgb_to_model(img) for c, img in obs.images.items()}
    pair = self.ts.push_frames(frames)
    desire = self.ts.push_desire(obs.desire)
    feats = self.ts.features()
    dev = self.device
    img = torch.from_numpy(pair["road"][None]).to(dev).float()
    big = torch.from_numpy(pair["wide"][None]).to(dev).float()
    hist = torch.from_numpy(feats[None]).to(dev)
    des = torch.from_numpy(desire[None]).to(dev)
    tc = torch.tensor([[1.0, 0.0]], device=dev)
    m = self.model
    base = m.base if hasattr(m, "base") else m
    v = base.encode(img, big)
    fb_full = torch.cat([hist, v.hidden[:, None]], 1)
    plan, _ = base.policy_raw(fb_full, des, tc)
    p = plan.reshape(1, 2, 33, 15)
    mu = op_to_ego_plan(p[0, 0].cpu().numpy())
    std = np.exp(p[0, 1].clamp(-7, 7).cpu().numpy())
    self.ts.push_hidden(v.hidden[0].cpu().numpy())
    steer, accel = self.ctrl(mu, obs.state.vx)
    info = {"plan": mu, "plan_std": std}
    if self.bev:
      if self.cones == "img" and getattr(m, "img", None) is not None:
        from fsop.eval.img_to_bev import bev_from_image_head

        xy, cls, sc = bev_from_image_head(m.img(v.s4, v.s8, v.s16), env.T_ego_cam())
      else:
        from fsop.eval.bev_metrics import decode_bev

        o = m.bev(v.s16, v.final)
        xy, cls, sc = decode_bev(o["heat"], o["off"])[0]
      info.update(cones_xy=xy, cones_cls=cls, cones_score=sc)
    return steer, mission_brake(accel, obs, env), info


def mission_brake(accel: float, obs: Obs, env: FSEnv, decel: float = 3.5, launch_accel: float = 2.0) -> float:
  """Mission layer (the AS state machine around the model):
  * launch: after "go" (AS Driving) a standing car gets at least ``launch_accel`` until it rolls - a still
    frame pair carries no information about whether to start (staging vs. finished look identical);
  * finish: after the last lap, brake on a constant-deceleration profile into the stop zone.
  Uses the sim's progress as a stand-in for an on-car lap counter + odometry (the rules require the car
  to count laps itself, D 8.3.4)."""
  rem = env.mission.s_end - obs.s
  v = obs.state.vx
  if v < 2.0 and rem > 15.0:
    accel = max(accel, launch_accel)
  if rem > 60.0:
    return accel
  v_allowed = float(np.sqrt(2 * decel * max(rem, 0.0)))
  if v > v_allowed:
    accel = min(accel, -(v * v - v_allowed**2) / max(2 * max(rem, 0.5), 1.0) - 0.5)
  if rem <= 0.5:
    accel = min(accel, -3.0 if v > 0.3 else -1.0)
  return accel


class GeometricPolicy:
  """Cone midline planner. ``source="gt"``: visible ground-truth cones (perfect perception);
  ``source="model"``: BEV detections of a fine-tuned checkpoint (modular perception -> planning)."""

  def __init__(self, source: str = "gt", model_policy: ModelPolicy | None = None, controller: str = "pp"):
    self.source, self.mp, self.controller_name, self.ctrl = source, model_policy, controller, None

  def reset(self) -> None:
    self.ctrl = None
    if self.mp is not None:
      self.mp.reset()

  def __call__(self, obs: Obs, env: FSEnv) -> tuple[float, float, dict]:
    from fsop.data.targets import visible_cones, world_to_ego
    from fsop.eval.geometric import plan_from_cones
    from fsop.track.schema import CONE_CLASS_INDEX

    if self.ctrl is None:
      self.ctrl = make_controller(self.controller_name, env.vp)
    if self.source == "gt":
      up = ~env.referee.down
      xy = world_to_ego(env.cones_xy[up], obs.state.x, obs.state.y, obs.state.yaw)
      cls = np.array([CONE_CLASS_INDEX.get(c, -1) for c, u in zip(env.cone_colors, up, strict=True) if u])
      vis = visible_cones(xy, env.T_ego_cam())
      xy, cls, info = xy[vis], cls[vis], {}
    else:
      _, _, info = self.mp(obs, env)
      xy, cls = info["cones_xy"], info["cones_cls"]
    plan = plan_from_cones(xy, cls, obs.state.vx)
    steer, accel = self.ctrl(plan, obs.state.vx)
    info = {**info, "plan": plan, "cones_xy": xy, "cones_cls": cls}
    return steer, mission_brake(accel, obs, env), info


@dataclass
class SuiteCfg:
  tracks: str = "data/tracks/td_v1"
  split: str = "test"
  n_trackdrive: int = 10
  laps: int = 2  # trackdrive laps in the suite (10 in competition; 2 keeps the suite fast)
  skidpad: bool = True
  acceleration: bool = True
  dr_levels: list[str] = field(default_factory=lambda: ["off", "light", "heavy"])
  seeds: list[int] = field(default_factory=lambda: [0])
  max_seconds: float = 180.0


def _suite_tracks(cfg: SuiteCfg):
  from fsop.track.batch import split_ids
  from fsop.track.fixed_layouts import acceleration, skidpad
  from fsop.track.io import load_json

  idx = json.loads((Path(cfg.tracks) / "index.json").read_text())["tracks"]
  ids = set(split_ids(idx)[cfg.split])
  out = []
  for r in idx:
    if r["id"] in ids and len([t for t in out if t.discipline == "trackdrive"]) < cfg.n_trackdrive:
      t = load_json(Path(cfg.tracks) / r["path"])
      t.laps = cfg.laps
      out.append(t)
  if cfg.skidpad:
    out.append(skidpad())
  if cfg.acceleration:
    out.append(acceleration())
  return out


def run_suite(suite_path: str | Path, ckpt: str | Path | None, out_dir: str | Path, controller: str = "pp",
              save_rrd: bool = True, baseline: str = "model") -> dict:
  from fsop.sim.drive import expert_policy, run_episode
  from fsop.sim.env import CAMERAS
  from fsop.sim.render.panda3d_backend import Panda3DRenderer
  from fsop.viz import rr_log

  suite_path = Path(suite_path)
  raw = yaml.safe_load(suite_path.read_text()) if suite_path.exists() else {}
  cfg = SuiteCfg(**raw)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  renderer = Panda3DRenderer(CAMERAS)
  if baseline == "expert":
    policy = expert_policy
  elif baseline == "geometric_gt":
    policy = GeometricPolicy("gt", controller=controller)
  elif baseline == "geometric_model":
    policy = GeometricPolicy("model", ModelPolicy.from_checkpoint(ckpt, controller), controller)
  else:
    policy = ModelPolicy.from_checkpoint(None if baseline == "openpilot" else ckpt, controller)
  results = []
  for track in _suite_tracks(cfg):
    for dr in cfg.dr_levels:
      for seed in cfg.seeds:
        sim = SimCfg(dr=dr, max_time=cfg.max_seconds)
        sim.noise.enabled = False
        env = FSEnv(track, sim, seed=seed, renderer=renderer)
        if hasattr(policy, "reset"):
          policy.reset()
        name = f"{track.id}_{dr}_s{seed}"
        if save_rrd:
          rr_log.init("fsop_closed_loop", save=str(out_dir / f"{name}.rrd"), recording_id=name,
                      blueprint=rr_log.sim_blueprint(bev=baseline == "model"))
        lat = []
        summ = run_episode(env, policy, viz=save_rrd, add_noise=False,
                           on_frame=lambda o, e, i, lat=lat: lat.append(abs(o.lat)))
        summ.update(track=track.id, dr=dr, seed=seed, max_lat=float(np.max(lat)) if lat else None,
                    rms_lat=float(np.sqrt(np.mean(np.square(lat)))) if lat else None,
                    progress=float(env.s / env.mission.s_end))
        results.append(summ)
        print(f"{name}: {summ['result']} laps {summ['n_laps']} doo {summ['doo']} oc {summ['oc']} "
              f"progress {summ['progress']:.2f} time {summ['time']}")
  agg = aggregate(results)
  report = {"ckpt": str(ckpt), "baseline": baseline, "controller": controller, "suite": raw, "runs": results,
            "aggregate": agg}
  (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
  (out_dir / "report.md").write_text(markdown_report(report))
  return report


def aggregate(results: list[dict]) -> dict:
  out = {}
  for d in sorted({r["discipline"] for r in results}):
    for dr in sorted({r["dr"] for r in results}):
      rs = [r for r in results if r["discipline"] == d and r["dr"] == dr]
      if not rs:
        continue
      out[f"{d}/{dr}"] = {
        "runs": len(rs),
        "pass_rate": float(np.mean([r["pass"] for r in rs])),
        "completion": float(np.mean([min(r["progress"], 1.0) for r in rs])),
        "doo_per_run": float(np.mean([r["doo"] for r in rs])),
        "oc_per_run": float(np.mean([r["oc"] for r in rs])),
        "mean_time": float(np.mean([r["time"] for r in rs if r["time"] is not None])) if any(
          r["time"] is not None for r in rs) else None,
        "rms_lat": float(np.mean([r["rms_lat"] for r in rs if r["rms_lat"] is not None])),
      }
  return out


def markdown_report(rep: dict) -> str:
  lines = [f"# Closed-loop report: {rep['baseline']} ({rep['ckpt']})", "",
           "| discipline / DR | runs | pass | completion | DOO/run | OC/run | mean time [s] | RMS lat [m] |",
           "|---|---|---|---|---|---|---|---|"]
  for k, a in rep["aggregate"].items():
    t = f"{a['mean_time']:.1f}" if a["mean_time"] is not None else "-"
    lines.append(f"| {k} | {a['runs']} | {a['pass_rate']:.0%} | {a['completion']:.0%} | {a['doo_per_run']:.1f} | "
                 f"{a['oc_per_run']:.1f} | {t} | {a['rms_lat']:.2f} |")
  return "\n".join(lines) + "\n"
