"""Open-loop evaluation on held-out sim tracks.

Vision is batched over all frames of an episode (it does not depend on history), then the policy runs
with the true temporal history computed by the *evaluated* model (never a stale cache).
Metrics (ego frame): longitudinal / lateral error at 0.5/1/2/4 s, ADE over 0-4 s, heading error, speed
error, Laplace calibration; BEV cone P/R/F1 per class and range; per discipline and DR level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fsop.data.sim_dataset import Episode, split_episodes
from fsop.data.targets import bev_targets
from fsop.eval.bev_metrics import BevScorer, decode_bev, gt_from_targets
from fsop.geometry.op_frames import op_to_ego_plan
from fsop.model.temporal import N_FEAT
from fsop.sim.expert import T_IDXS
from fsop.train.device import pick_device

HORIZONS = (0.5, 1.0, 2.0, 4.0)


@torch.no_grad()
def run_episode_model(model, ep: Episode, device, batch: int = 32, bev: bool = True) -> dict:
  base = model.base if hasattr(model, "base") else model
  road, wide = ep.frames["road"][:], ep.frames["wide"][:]
  T = ep.T
  hidden = torch.zeros(T, 512, device=device)
  heats, offs, img_cones = [], [], []
  has_img = getattr(model, "img", None) is not None
  for a in range(0, T, batch):
    ks = np.arange(a, min(a + batch, T))
    k0 = np.maximum(ks - 1, 0)
    img = torch.from_numpy(np.concatenate([road[k0], road[ks]], 1)).to(device).float()
    big = torch.from_numpy(np.concatenate([wide[k0], wide[ks]], 1)).to(device).float()
    v = base.encode(img, big)
    hidden[ks] = v.hidden
    if bev and hasattr(model, "bev"):
      o = model.bev(v.s16, v.final)
      heats.append(o["heat"].cpu())
      offs.append(o["off"].cpu())
    if has_img:
      from fsop.eval.img_to_bev import bev_from_image_head

      io_ = model.img(v.s4, v.s8, v.s16)
      img_cones.extend(bev_from_image_head(io_, ep.T_ego_cam, b=j) for j in range(len(ks)))
  mus, stds = [], []
  tc = torch.tensor([[1.0, 0.0]], device=device)
  desire = torch.from_numpy(ep.desire).to(device)
  for a in range(0, T, batch):
    ks = np.arange(a, min(a + batch, T))
    hist = torch.zeros(len(ks), N_FEAT + 1, 512, device=device)
    des = torch.zeros(len(ks), N_FEAT + 1, 8, device=device)
    for j, k in enumerate(ks):
      idx = np.arange(k - N_FEAT, k + 1)
      ok = idx >= 0
      hist[j, torch.from_numpy(ok)] = hidden[idx[ok]]
      des[j, torch.from_numpy(ok)] = desire[idx[ok]]
    plan, _ = base.policy_raw(hist, des, tc.expand(len(ks), -1))
    p = plan.reshape(len(ks), 2, 33, 15)
    mus.append(p[:, 0].cpu().numpy())
    stds.append(np.exp(p[:, 1].clamp(-7, 7).cpu().numpy()))
  out = {"mu": op_to_ego_plan(np.concatenate(mus)), "std": np.concatenate(stds) * 1.0}
  if heats:
    out["heat"], out["off"] = torch.cat(heats), torch.cat(offs)
  if img_cones:
    out["img_cones"] = img_cones
  return out


def plan_errors(mu: np.ndarray, gt: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
  e = {}
  for t in HORIZONS:
    i = int(np.argmin(np.abs(T_IDXS - t)))
    e[f"long_{t}s"] = np.abs(mu[:, i, 0] - gt[:, i, 0])
    e[f"lat_{t}s"] = np.abs(mu[:, i, 1] - gt[:, i, 1])
  m = T_IDXS <= 4.0
  e["ade_4s"] = np.linalg.norm(mu[:, m, :2] - gt[:, m, :2], axis=2).mean(1)
  i1 = int(np.argmin(np.abs(T_IDXS - 1.0)))
  e["heading_1s_deg"] = np.rad2deg(np.abs(mu[:, i1, 11] - gt[:, i1, 11]))
  i05 = int(np.argmin(np.abs(T_IDXS - 0.5)))
  e["speed_0.5s"] = np.abs(mu[:, i05, 3] - gt[:, i05, 3])
  # calibration: fraction of lateral residuals inside +-1 b at 2 s (Laplace: ideal 63 %)
  i2 = int(np.argmin(np.abs(T_IDXS - 2.0)))
  e["calib_lat_2s"] = (np.abs(mu[:, i2, 1] - gt[:, i2, 1]) < std[:, i2, 1]).astype(float)
  return e


def evaluate(dataset: str | Path, ckpt: str | Path | None = None, split: str = "test", rrd: str | Path | None = None,
             max_episodes: int = 0, min_k: int = 5) -> dict:
  from fsop.model.build import build_base, load_fs

  device = pick_device()
  model = (build_base(None) if ckpt is None else load_fs(ckpt)).to(device).eval()
  ids = split_episodes(dataset, split)
  if max_episodes:
    ids = ids[:max_episodes]
  per: dict[str, dict[str, list]] = {}
  scorers: dict[str, BevScorer] = {}
  if rrd:
    from fsop.viz import rr_log

    rr_log.init("fsop_open_loop", save=str(rrd))
  for eid in tqdm(ids, desc="open-loop"):
    ep = Episode(Path(dataset) / eid)
    r = run_episode_model(model, ep, device)
    gt = ep.plan
    errs = plan_errors(r["mu"][min_k:], gt[min_k:], r["std"][min_k:])
    keys = ["all", ep.meta["discipline"], f"dr_{ep.meta['dr']['level']}"]
    for key in keys:
      d = per.setdefault(key, {})
      for k, v in errs.items():
        d.setdefault(k, []).append(v)
    if "heat" in r:
      for k in range(min_k, ep.T):
        cones, cls = ep.cones_ego(k)
        heat, off, pos, _ = bev_targets(cones, cls, ep.bev_mask)
        (pxy, pc, _), = decode_bev(r["heat"][k:k + 1], r["off"][k:k + 1])
        gxy, gc = gt_from_targets(torch.from_numpy(heat[None]), torch.from_numpy(off[None]),
                                  torch.from_numpy(pos[None]))[0]
        for key in keys:
          scorers.setdefault(key, BevScorer()).add(pxy, pc, gxy, gc)
          if "img_cones" in r:
            ixy, ic, _ = r["img_cones"][k]
            scorers.setdefault(key + "/imgproj", BevScorer()).add(ixy, ic, gxy, gc)
    if rrd:
      _log_episode(ep, r, eid)
  summary = {}
  for key, d in per.items():
    s = {k: float(np.mean(np.concatenate(v))) for k, v in d.items()}
    for sk, prefix in ((key, "bev"), (key + "/imgproj", "bev_imgproj")):
      if sk in scorers:
        b = scorers[sk].summary()
        s.update({f"{prefix}_f1": b["f1"], f"{prefix}_precision": b["precision"], f"{prefix}_recall": b["recall"],
                  f"{prefix}_pos_err": b["pos_err_m"], f"{prefix}_swap": b["blue_yellow_swap_rate"],
                  f"{prefix}_by_range": b["by_range"]})
    s["n_frames"] = int(sum(len(x) for x in d["ade_4s"]))
    summary[key] = s
  return {"ckpt": str(ckpt), "split": split, "episodes": ids, "summary": summary}


def _log_episode(ep: Episode, r: dict, eid: str) -> None:
  import rerun as rr

  from fsop.geometry.camera import se2_to_se3
  from fsop.geometry.yuv import model_to_rgb
  from fsop.track.io import load_json
  from fsop.viz import rr_log

  track = load_json(ep.path / "track.json")
  base = f"episodes/{eid}"
  rr_log.log_track(track, ep.cones_xy, _colors(ep), path=f"{base}/world/track")
  for k in range(ep.T):
    rr.set_time("frame", sequence=k)
    x, y, yaw = ep.pose(k)
    rr_log.log_ego(se2_to_se3(x, y, yaw), path=f"{base}/world/ego")
    rr_log.log_plan("gt", ep.plan[k], (40, 220, 80), path=f"{base}/world/ego/plan")
    rr_log.log_plan("pred", r["mu"][k], (230, 60, 200), r["std"][k], path=f"{base}/world/ego/plan")
    if k % 2 == 0:
      rr.log(f"{base}/wide", rr.Image(model_to_rgb(ep.frames["wide"][k])).compress(jpeg_quality=80))


def _colors(ep: Episode):
  from fsop.track.schema import CONE_CLASSES, ConeColor

  return [CONE_CLASSES[c] if 0 <= c < len(CONE_CLASSES) else ConeColor.UNKNOWN for c in ep.cones_cls]
