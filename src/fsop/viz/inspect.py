"""Replay a recorded episode in Rerun: camera frames with projected GT cones + GT plan (and model
predictions when a checkpoint is given), BEV targets vs predictions, telemetry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

from fsop.data.sim_dataset import Episode
from fsop.data.targets import bev_targets
from fsop.geometry.camera import se2_to_se3
from fsop.geometry.yuv import model_to_rgb
from fsop.track.io import load_json
from fsop.track.schema import CONE_CLASSES, ConeColor
from fsop.viz import rr_log


def inspect_episode(path: str | Path, save: str | Path | None = None, ckpt: str | Path | None = None,
                    every: int = 1) -> None:
  ep = Episode(path)
  rr_log.init("fsop_episode", spawn=save is None, save=str(save) if save else None,
              blueprint=rr_log.sim_blueprint(bev=True))
  colors = [CONE_CLASSES[c] if 0 <= c < len(CONE_CLASSES) else ConeColor.UNKNOWN for c in ep.cones_cls]
  rr_log.log_track(load_json(ep.path / "track.json"), ep.cones_xy, colors)
  rr_log.log_cameras(ep.T_ego_cam)
  pred = None
  if ckpt is not None:
    import torch

    from fsop.eval.open_loop import run_episode_model
    from fsop.model.build import load_fs
    from fsop.train.device import pick_device

    dev = pick_device()
    pred = run_episode_model(load_fs(ckpt).to(dev).eval(), ep, dev)
    _ = torch
  for k in range(0, ep.T, every):
    rr.set_time("frame", sequence=k)
    rr.set_time("sim_time", duration=float(ep.states[k, 0]))
    x, y, yaw = ep.pose(k)
    rr_log.log_ego(se2_to_se3(x, y, yaw))
    rr_log.log_images({c: model_to_rgb(ep.frames[c][k]) for c in ("road", "wide")})
    rr_log.log_plan("gt", ep.plan[k], (40, 220, 80))
    cones, cls = ep.cones_ego(k)
    heat, *_ = bev_targets(cones, cls, ep.bev_mask)
    rr.log("bev/gt", rr.Image(rr_log.bev_image(heat)))
    if pred is not None:
      rr_log.log_plan("pred", pred["mu"][k], (230, 60, 200), pred["std"][k])
      if "heat" in pred:
        from fsop.eval.bev_metrics import decode_bev

        p = 1 / (1 + np.exp(-pred["heat"][k].numpy()))
        rr.log("bev/pred", rr.Image(rr_log.bev_image(p)))
        (xy, c, s), = decode_bev(pred["heat"][k:k + 1], pred["off"][k:k + 1])
        rr_log.log_cones_ego("cones_pred", xy, c, s)
    st = ep.states[k]
    rr_log.log_scalars("metrics", {"speed": st[4], "steer_deg": np.rad2deg(st[7]), "lat_err": st[10],
                                   "noise_deg": np.rad2deg(st[11])})
