"""Team profiles: everything that is specific to *your* car, kept out of the code.

A profile (``configs/profiles/*.yaml``) describes
  * ``vehicle``  - geometry, mass, tyres, actuator limits (overrides of ``fsop.sim.vehicle.VehicleParams``)
  * ``mount``    - nominal camera position/orientation on the car (``fsop.geometry.camera.CameraMount``,
                   angles in degrees); simulation domain randomization jitters around it
  * ``cameras``  - the physical cameras used on the real car and which model input slot ("road",
                   "wide") each feeds; used by ``fsop.geometry.warp`` to map real images into
                   openpilot's model frame (one camera may feed both slots)
Every simulator / evaluation / runtime entry point accepts ``--profile``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from fsop.geometry.camera import CameraMount
from fsop.geometry.mounting import MountDynamics
from fsop.sim.vehicle import VehicleParams

DEFAULT_PROFILE = Path("configs/profiles/default.yaml")


@dataclass
class CameraSpec:
  name: str = "front"
  width: int = 1928
  height: int = 1208
  fx: float = 2648.0
  fy: float = 2648.0
  cx: float = 964.0
  cy: float = 604.0
  dist: list[float] = field(default_factory=list)  # OpenCV k1, k2, p1, p2[, k3] or fisheye k1..k4
  fisheye: bool = False
  slots: list[str] = field(default_factory=lambda: ["road", "wide"])
  # orientation of this camera relative to the car (deg); position is taken from ``mount``
  roll_deg: float = 0.0
  pitch_deg: float = 0.0
  yaw_deg: float = 0.0

  @property
  def K(self) -> np.ndarray:
    return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


@dataclass
class TeamProfile:
  name: str = "default"
  team: str = ""
  vehicle: VehicleParams = field(default_factory=VehicleParams)
  mount: CameraMount = field(default_factory=lambda: CameraMount(z=1.1))
  cameras: list[CameraSpec] = field(default_factory=lambda: [CameraSpec()])
  notes: str = ""
  motion: MountDynamics = field(default_factory=MountDynamics)
  calibration_error: CameraMount = field(default_factory=lambda: CameraMount(z=0.0))

  def __post_init__(self):
    v = self.vehicle
    positive = ("mass", "iz", "lf", "lr", "track", "length", "width", "cf", "cr", "mu", "max_speed",
                "max_steer", "max_steer_rate", "max_accel", "max_brake", "tau_steer", "tau_accel")
    if any(not np.isfinite(getattr(v, k)) or getattr(v, k) <= 0 for k in positive):
      raise ValueError("Vehicle geometry and dynamic limits must be finite and positive")
    if v.v_blend[0] < 0 or v.v_blend[1] <= v.v_blend[0]:
      raise ValueError("Invalid kinematic/dynamic blending interval")
    if not np.isfinite(list(asdict(self.mount).values())).all() or self.mount.z <= 0:
      raise ValueError("Camera installation must be finite and above the ground")
    if len({c.name for c in self.cameras}) != len(self.cameras):
      raise ValueError("Physical camera names must be unique")
    for c in self.cameras:
      if min(c.width, c.height, c.fx, c.fy) <= 0 or not np.isfinite(c.K).all():
        raise ValueError("Invalid camera resolution or intrinsics")
    for slot in ("road", "wide"):
      if sum(slot in c.slots for c in self.cameras) != 1:
        raise ValueError(f"Exactly one physical camera must feed {slot}")

  def camera_for(self, slot: str) -> CameraSpec:
    for c in self.cameras:
      if slot in c.slots:
        return c
    raise KeyError(f"profile {self.name!r} has no camera feeding the {slot!r} slot")

  def to_dict(self) -> dict:
    d = asdict(self)
    m = d["mount"]
    for k in ("roll", "pitch", "yaw"):
      m[f"{k}_deg"] = float(np.rad2deg(m.pop(k)))
    for k in ("roll", "pitch", "yaw"):
      d["calibration_error"][f"{k}_deg"] = float(np.rad2deg(d["calibration_error"].pop(k)))
    d["vehicle"]["max_steer_deg"] = float(np.rad2deg(d["vehicle"].pop("max_steer")))
    d["vehicle"]["max_steer_rate_deg"] = float(np.rad2deg(d["vehicle"].pop("max_steer_rate")))
    return d


def _pick(cls, d: dict) -> dict:
  names = {f.name for f in fields(cls)}
  unknown = set(d) - names
  if unknown:
    raise ValueError(f"unknown {cls.__name__} keys in profile: {sorted(unknown)}")
  return d


def load_profile(path: str | Path | None = None) -> TeamProfile:
  """Load a profile yaml (``None`` -> built-in defaults, which match configs/profiles/default.yaml)."""
  if path is None or str(path) == "":
    return TeamProfile()
  raw = yaml.safe_load(Path(path).read_text()) or {}
  return profile_from_dict(raw, Path(path).stem)


def profile_from_dict(raw: dict, name: str = "profile") -> TeamProfile:
  """Decode a serialized profile with the same validation as a YAML file."""
  veh = dict(raw.get("vehicle", {}))
  for k in ("max_steer", "max_steer_rate"):
    if f"{k}_deg" in veh:
      veh[k] = float(np.deg2rad(veh.pop(f"{k}_deg")))
  if "v_blend" in veh:
    veh["v_blend"] = tuple(veh["v_blend"])
  mount = dict(raw.get("mount", {}))
  for k in ("roll", "pitch", "yaw"):
    if f"{k}_deg" in mount:
      mount[k] = float(np.deg2rad(mount.pop(f"{k}_deg")))
  cams = [CameraSpec(**_pick(CameraSpec, c)) for c in raw.get("cameras", [])] or [CameraSpec()]
  error = dict(raw.get("calibration_error", {}))
  error.setdefault("z", 0.0)
  for k in ("roll", "pitch", "yaw"):
    if f"{k}_deg" in error:
      error[k] = float(np.deg2rad(error.pop(f"{k}_deg")))
  return TeamProfile(motion=MountDynamics(**_pick(MountDynamics, raw.get("motion", {}))),
                     calibration_error=CameraMount(**_pick(CameraMount, error)),
                     name=raw.get("name", name), team=raw.get("team", ""),
                     vehicle=VehicleParams(**_pick(VehicleParams, veh)),
                     mount=CameraMount(**_pick(CameraMount, mount)), cameras=cams, notes=raw.get("notes", ""))
