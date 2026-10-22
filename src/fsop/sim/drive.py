"""Run one episode (expert or a policy) with live/recorded Rerun visualization."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fsop.sim.env import FSEnv, Obs
from fsop.viz import rr_log

Policy = Callable[[Obs, FSEnv], tuple[float, float, dict]]


def expert_policy(obs: Obs, env: FSEnv) -> tuple[float, float, dict]:
  s, a = env.expert_action()
  return s, a, {}


def run_episode(env: FSEnv, policy: Policy = expert_policy, viz: bool = True, label_every: int = 4,
                add_noise: bool = True, max_frames: int | None = None, on_frame=None) -> dict:
  import rerun as rr

  obs = env.observe()
  if viz:
    rr_log.log_track(env.track, env.cones_xy, env.cone_colors)
    rr_log.log_ego(env.T_world_ego(), env.vp)
    rr_log.log_cameras(env.T_ego_cam(), env.cfg.cameras)
  while not env.done:
    steer, accel, info = policy(obs, env)
    if viz:
      rr.set_time("frame", sequence=obs.frame)
      rr.set_time("sim_time", duration=obs.t)
      rr_log.log_ego(obs.T_world_ego)
      if obs.images:
        rr_log.log_images(obs.images)
      if obs.frame % label_every == 0:
        rr_log.log_plan("gt", env.label(), (40, 220, 80))
      if "plan" in info:
        rr_log.log_plan("pred", info["plan"], (230, 60, 200), info.get("plan_std"))
      if "cones_xy" in info:
        rr_log.log_cones_ego("cones_pred", info["cones_xy"], info["cones_cls"], info.get("cones_score"))
      rr_log.log_scalars("metrics", {"speed": obs.state.vx, "steer_deg": np.rad2deg(obs.state.steer),
                                     "lat_err": obs.lat, "noise_deg": np.rad2deg(obs.noise),
                                     "yaw_rate": obs.state.r})
      rr_log.log_events(obs.events)
      if any(e.kind == "doo" for e in obs.events):
        down = env.cones_xy[env.referee.down]
        rr.log("world/track/cones_down", rr.Points3D(np.c_[down, np.full(len(down), 0.05)], colors=(255, 0, 0),
                                                     radii=0.2))
    if on_frame is not None:
      on_frame(obs, env, info)
    obs = env.step(steer, accel, add_noise=add_noise)
    if max_frames is not None and obs.frame >= max_frames:
      break
  summary = env.referee.summary()
  if viz:
    import json

    rr.log("race/summary", rr.TextDocument("```json\n" + json.dumps(summary, indent=1, default=float) + "\n```",
                                           media_type="text/markdown"), static=True)
  return summary
