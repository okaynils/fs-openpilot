"""FS driverless simulation environment.

Loop at 20 Hz (model rate) with 5 x 100 Hz dynamics substeps:
  obs(t) = render(state_t)  ->  control  ->  integrate  ->  referee
Labels (counterfactual expert plans) are computed from state_t by ``label()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from fsop.geometry.camera import CAMERA_MODELS, CameraModel, CameraMount, se2_to_se3
from fsop.sim.expert import Expert, ExpertParams, Mission, plan_from_rollout, rollout, sample_expert
from fsop.sim.randomization import DRCfg, SceneDR, dr_preset, postprocess, sample_scene
from fsop.sim.referee import Event, Referee
from fsop.sim.vehicle import VehicleParams, VehicleState, step
from fsop.track.schema import ConeColor, Track

DESIRE_LEN = 8
# skidpad route commands re-use openpilot desire channels: 1 = turnLeft, 2 = turnRight, 7 = straight/exit
ROUTE_DESIRE = {"left": 1, "right": 2, "exit": 7}


@dataclass
class NoiseCfg:
  enabled: bool = True
  steer_sigma: float = 0.05  # [rad / sqrt(s)] OU diffusion
  steer_theta: float = 1.5
  on_time: tuple[float, float] = (1.5, 5.0)
  off_time: tuple[float, float] = (2.0, 7.0)
  speed_scale: tuple[float, float] = (0.85, 1.1)
  # recovery excursions: during a noise burst the expert drives a line offset from the planned path
  # (|offset| in [offset_min, offset_max] m, probability offset_prob per burst) and returns to the reference
  # when the burst ends; the labels stay the unperturbed expert, so these frames teach recovery.
  offset_max: float = 0.0
  offset_min: float = 0.3
  offset_prob: float = 0.6
  offset_tau: float = 0.6  # [s] first-order approach to the excursion target


@dataclass
class SimCfg:
  physical_camera: bool = False  # render sensor pixels then use deployment remap
  fps: int = 20
  substeps: int = 5
  max_time: float = 0.0  # 0 = derived from mission length (at 2.5 m/s) + 60 s
  render: bool = True
  cameras: tuple[str, ...] = ("road", "wide")
  dr: str = "light"
  noise: NoiseCfg = field(default_factory=NoiseCfg)
  randomize_expert: bool = True
  label_dt: float = 0.025
  start_speed: float = 0.0
  random_start: bool = False  # closed tracks: start at a random point with a random fraction of v_ref
  venue_prob: float = 0.0  # probability of a venue-realism scene (DRCfg.venue_prob)


CAMERAS: dict[str, CameraModel] = CAMERA_MODELS


@dataclass
class Obs:
  frame: int
  t: float
  state: VehicleState
  s: float
  lat: float
  images: dict[str, np.ndarray]  # RGB model-frame images
  T_world_ego: np.ndarray
  desire: np.ndarray  # [8] pulse
  events: list[Event]
  noise: float


class FSEnv:
  def __init__(self, track: Track, cfg: SimCfg | None = None, seed: int = 0, renderer=None,
               vp: VehicleParams | None = None, expert_params: ExpertParams | None = None,
               dr_cfg: DRCfg | None = None, profile=None):
    from fsop.profile import TeamProfile

    self.profile = profile or TeamProfile()
    self.track = track
    self.cfg = cfg or SimCfg()
    self.seed = seed
    self.rng = np.random.default_rng(seed)
    self.vp = vp or self.profile.vehicle
    self.dr_cfg = dr_cfg or dr_preset(self.cfg.dr)
    if self.cfg.venue_prob > 0:
      self.dr_cfg.venue_prob = self.cfg.venue_prob
    self.scene: SceneDR = sample_scene(self.rng, self.dr_cfg)
    m = self.profile.mount  # nominal team mount + per-episode DR perturbation
    self.mount = CameraMount(x=m.x, y=m.y, z=m.z + self.scene.cam_dz, roll=m.roll + self.scene.cam_roll,
                             pitch=m.pitch + self.scene.cam_pitch, yaw=m.yaw + self.scene.cam_yaw)
    if expert_params is None:
      expert_params = sample_expert(self.rng, track.discipline) if self.cfg.randomize_expert else ExpertParams()
    expert_params = replace(expert_params, v_cap=min(expert_params.v_cap, self.vp.max_speed),
                            a_acc=min(expert_params.a_acc, self.vp.max_accel),
                            a_brk=min(expert_params.a_brk, self.vp.max_brake),
                            a_lat_max=min(expert_params.a_lat_max, self.vp.mu * 9.81),
                            stop_decel=min(expert_params.stop_decel, self.vp.max_brake))
    self.expert_params = expert_params
    self.cones_xy, self.cone_colors, self.cone_src = self._episode_cones()
    self.mission = Mission(track, expert_params, self.vp, cones_xy=self.cones_xy)
    self.expert = Expert(self.mission, self.vp)
    self.warper = None
    self.render_cameras = {k: CAMERAS[k] for k in self.cfg.cameras}
    if self.cfg.physical_camera:
      from fsop.geometry.warp import ModelFrameWarper
      self.warper = ModelFrameWarper(self.profile)
      self.render_cameras = {c.name: CameraModel(c.name, c.K, c.width, c.height) for c in self.profile.cameras}
      if any(c.dist or c.fisheye for c in self.profile.cameras):
        raise ValueError("Physical simulation currently supports rectilinear undistorted sensors only")
    self.renderer = renderer
    if self.cfg.render and self.renderer is None:
      from fsop.sim.render.panda3d_backend import Panda3DRenderer

      self.renderer = Panda3DRenderer(self.render_cameras)
    self.reset()

  # ------------------------------------------------------------------------------------------
  def _episode_cones(self):
    xy, colors, src = [], [], []
    d = self.dr_cfg
    for i, c in enumerate(self.track.cones):
      if c.role in ("left", "right") and self.rng.random() < d.cone_missing:
        continue
      j = self.rng.normal(0, d.cone_jitter, 2) if d.cone_jitter > 0 else np.zeros(2)
      xy.append([c.x + j[0], c.y + j[1]])
      colors.append(c.color)
      src.append(i)
    if d.cone_extra > 0 and self.track.closed:
      n_extra = self.rng.poisson(d.cone_extra * self.track.length / 100)
      left, right = self.track.boundaries()
      for _ in range(n_extra):
        k = self.rng.integers(len(left))
        side = left if self.rng.random() < 0.5 else right
        out = side[k] + (side[k] - self.track.centerline[k]) * self.rng.uniform(0.3, 1.5)
        xy.append(out.tolist())
        colors.append(ConeColor.ORANGE if self.rng.random() < 0.5 else ConeColor.UNKNOWN)
        src.append(-1)
    return np.asarray(xy, dtype=float).reshape(-1, 2), colors, np.array(src)

  def reset(self) -> Obs:
    x, y, yaw = self.track.start_pose
    v0 = self.cfg.start_speed
    if self.cfg.random_start and self.track.closed:
      s0 = float(self.rng.uniform(0, self.mission.L))
      p0, p1 = self.mission.point_at(s0), self.mission.point_at(s0 + 0.5)
      lat0 = float(self.rng.normal(0, 0.3))
      h = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
      x, y = p0[0] - np.sin(h) * lat0, p0[1] + np.cos(h) * lat0
      yaw = h + float(self.rng.normal(0, 0.05))
      v0 = float(self.mission.v_at(s0) * self.rng.uniform(0.5, 1.0))
    self.state = VehicleState(x, y, yaw, vx=v0)
    self.hint, self.s = self.mission.init_progress(x, y)
    _, _, self.lat = self.mission.update_progress(x, y, self.hint, self.s)
    self.referee = Referee(self.track, self.cones_xy, self.cone_colors, self.vp)
    self.frame = 0
    self.noise_val = 0.0
    self.noise_on = False
    self.noise_timer = float(self.rng.uniform(*self.cfg.noise.off_time))
    self.speed_scale = 1.0
    self.offset_val = 0.0
    self.offset_target = 0.0
    self.desire_sent: set[int] = set()
    self.pending_events: list[Event] = []
    if self.renderer is not None:
      self.renderer.load_scene(self.track, self.cones_xy, self.cone_colors, self.scene)
    return self.observe()

  @property
  def dt(self) -> float:
    return 1.0 / self.cfg.fps

  def T_world_ego(self) -> np.ndarray:
    return se2_to_se3(self.state.x, self.state.y, self.state.yaw)

  def T_ego_cam(self) -> np.ndarray:
    return self.profile.motion.pose(self.mount, self.state.t).T_ego_cam()

  def calibrated_mount(self) -> CameraMount:
    """Estimated installation available at deployment; flex is deliberately unobserved."""
    e = self.profile.calibration_error
    return CameraMount(**{k: getattr(self.mount, k) + getattr(e, k)
                          for k in ("x", "y", "z", "roll", "pitch", "yaw")})

  def mission_done(self) -> bool:
    return self.s >= self.mission.s_end - 0.5

  def route_desire(self) -> np.ndarray:
    d = np.zeros(DESIRE_LEN, np.float32)
    segs = self.track.generator.get("segments")
    if not segs:
      return d
    # one pulse ~2 s before entering each route segment (right circle, left circle, exit)
    v = max(self.state.vx, 3.0)
    for i, name in enumerate(("right", "left", "exit")):
      s_c = float(self.mission.s_arr[min(segs[name][0], self.mission.n - 1)])
      if 0 < s_c - self.s < 2.0 * v and i not in self.desire_sent:
        self.desire_sent.add(i)
        d[ROUTE_DESIRE[name]] = 1.0
    return d

  def observe(self, render: bool = True) -> Obs:
    imgs = {}
    self.sensor_images = {}  # latest physical RGB pixels, for end-to-end deployment evaluation
    if self.renderer is not None and render:
      T = self.T_world_ego() @ self.T_ego_cam()
      transforms = {k: T for k in self.render_cameras}
      if self.warper is not None:
        from fsop.geometry.warp import _cam_rot
        for c in self.profile.cameras:
          offset = np.eye(4)
          offset[:3, :3] = _cam_rot(*np.deg2rad([c.roll_deg, c.pitch_deg, c.yaw_deg]))
          transforms[c.name] = T @ offset
      raw = self.renderer.render(transforms)
      if self.vp.body_height > 0:
        from fsop.geometry.mounting import body_image_mask

        for k, image in raw.items():
          image[body_image_mask(self.render_cameras[k],
                                     np.linalg.inv(self.T_world_ego()) @ transforms[k], self.vp)] = (35, 38, 42)
      imgs = {k: postprocess(v, self.scene, self.rng, self.state.r) for k, v in raw.items()}
      if self.warper is not None:
        self.sensor_images = imgs
        imgs = self.warper(imgs)
    ev, self.pending_events = self.pending_events, []
    return Obs(self.frame, self.state.t, self.state.copy(), self.s, self.lat, imgs, self.T_world_ego(),
               self.route_desire(), ev, self.noise_val)

  # ------------------------------------------------------------------------------------------
  def expert_action(self) -> tuple[float, float]:
    return self.expert.act(self.state, self.s, self.offset_val)

  def _update_noise(self, dt: float) -> float:
    n = self.cfg.noise
    if not n.enabled:
      return 0.0
    self.noise_timer -= dt
    if self.noise_timer <= 0:
      self.noise_on = not self.noise_on
      self.noise_timer = float(self.rng.uniform(*(n.on_time if self.noise_on else n.off_time)))
      self.speed_scale = float(self.rng.uniform(*n.speed_scale)) if self.noise_on else 1.0
      self.offset_target = 0.0
      if self.noise_on and n.offset_max > 0 and self.rng.random() < n.offset_prob:
        self.offset_target = float(self.rng.choice((-1.0, 1.0)) * self.rng.uniform(n.offset_min, n.offset_max))
    self.offset_val += (self.offset_target - self.offset_val) * min(dt / n.offset_tau, 1.0)
    target_sigma = n.steer_sigma if self.noise_on else 0.0
    self.noise_val += -n.steer_theta * self.noise_val * dt + target_sigma * np.sqrt(dt) * self.rng.normal()
    if not self.noise_on:
      self.noise_val *= np.exp(-3.0 * dt)
    return self.noise_val

  def step(self, steer: float, accel: float, add_noise: bool = True, render: bool = True) -> Obs:
    dt = self.dt / self.cfg.substeps
    noise = self._update_noise(self.dt) if add_noise else 0.0
    if add_noise and self.noise_on:
      accel = accel + 0.8 * (self.speed_scale - 1.0) * max(self.state.vx, 1.0)
    for _ in range(self.cfg.substeps):
      self.state = step(self.state, steer + noise, accel, self.vp, dt)
    self.hint, self.s, self.lat = self.mission.update_progress(self.state.x, self.state.y, self.hint, self.s)
    ev = self.referee.update(self.state, self.mission_done())
    for e in ev:
      if e.kind == "doo" and self.renderer is not None:
        self.renderer.knock_cone(e.data["cone"], e.data["dir"])
    self.pending_events.extend(ev)
    self.frame += 1
    return self.observe(render)

  def label(self) -> np.ndarray:
    traj = rollout(self.mission, self.expert, self.vp, self.state, self.s, self.hint, dt=self.cfg.label_dt)
    return plan_from_rollout(traj, dt=self.cfg.label_dt)

  @property
  def max_time(self) -> float:
    if self.cfg.max_time > 0:
      return self.cfg.max_time
    return self.mission.s_end / 2.5 + 60.0

  @property
  def done(self) -> bool:
    return self.referee.done or self.state.t >= self.max_time
