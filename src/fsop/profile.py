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
  return TeamProfile(name=raw.get("name", Path(path).stem), team=raw.get("team", ""),
                     vehicle=VehicleParams(**_pick(VehicleParams, veh)),
                     mount=CameraMount(**_pick(CameraMount, mount)), cameras=cams, notes=raw.get("notes", ""))
