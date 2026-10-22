"""Track data model shared by the generator, simulator, dataset and visualization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Literal

import numpy as np

SCHEMA_VERSION = 1

Discipline = Literal["trackdrive", "autocross", "skidpad", "acceleration"]


class ConeColor(StrEnum):
  BLUE = "blue"
  YELLOW = "yellow"
  ORANGE = "orange"
  BIG_ORANGE = "big_orange"
  UNKNOWN = "unknown"


# Class indices used by every learned head (BEV + image) and by FSOCO conversion.
CONE_CLASSES: tuple[ConeColor, ...] = (ConeColor.BLUE, ConeColor.YELLOW, ConeColor.ORANGE, ConeColor.BIG_ORANGE)
CONE_CLASS_INDEX = {c: i for i, c in enumerate(CONE_CLASSES)}

# RGB used for visualization (not rendering).
CONE_VIZ_RGB = {
  ConeColor.BLUE: (40, 90, 230),
  ConeColor.YELLOW: (250, 210, 20),
  ConeColor.ORANGE: (255, 120, 10),
  ConeColor.BIG_ORANGE: (230, 70, 0),
  ConeColor.UNKNOWN: (160, 160, 160),
}


@dataclass
class Cone:
  x: float
  y: float
  color: ConeColor
  role: str = "boundary"  # left | right | start_gate | marking | stop_zone | skidpad_*
  id: int = -1


@dataclass
class TrackStats:
  length: float = 0.0
  min_radius: float = float("inf")  # centerline
  max_straight: float = 0.0
  min_width: float = 0.0
  n_cones: int = 0


@dataclass
class Track:
  id: str
  discipline: Discipline
  seed: int
  closed: bool
  centerline: np.ndarray  # [N,2] world, resampled at ds
  width_left: np.ndarray  # [N] distance centerline -> left boundary
  width_right: np.ndarray  # [N]
  cones: list[Cone] = field(default_factory=list)
  start_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, yaw where the car is staged
  timekeeping_line: tuple[tuple[float, float], tuple[float, float]] | None = None
  finish_line: tuple[tuple[float, float], tuple[float, float]] | None = None
  stop_zone: list[tuple[float, float]] | None = None  # polygon
  route: list[str] = field(default_factory=list)  # skidpad route segments
  laps: int = 1
  stats: TrackStats = field(default_factory=TrackStats)
  generator: dict = field(default_factory=dict)
  schema_version: int = SCHEMA_VERSION

  # ---- derived geometry -------------------------------------------------------------------
  @property
  def s(self) -> np.ndarray:
    d = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

  @property
  def length(self) -> float:
    s = self.s[-1]
    if self.closed:
      s += float(np.linalg.norm(self.centerline[0] - self.centerline[-1]))
    return float(s)

  def heading(self) -> np.ndarray:
    from fsop.track.centerline import headings

    return headings(self.centerline, self.closed)

  def curvature(self) -> np.ndarray:
    from fsop.track.centerline import curvature

    return curvature(self.centerline, self.closed)

  def boundaries(self) -> tuple[np.ndarray, np.ndarray]:
    from fsop.track.centerline import normals

    n = normals(self.centerline, self.closed)
    left = self.centerline + n * self.width_left[:, None]
    right = self.centerline - n * self.width_right[:, None]
    return left, right

  def cone_array(self) -> tuple[np.ndarray, np.ndarray]:
    """Returns cone positions [M,2] and class index [M] (-1 for classes outside CONE_CLASSES)."""
    if not self.cones:
      return np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)
    xy = np.array([[c.x, c.y] for c in self.cones], dtype=np.float64)
    cls = np.array([CONE_CLASS_INDEX.get(c.color, -1) for c in self.cones], dtype=np.int64)
    return xy, cls

  # ---- (de)serialization ----------------------------------------------------------------
  def to_dict(self) -> dict:
    d = {
      "id": self.id,
      "discipline": self.discipline,
      "seed": self.seed,
      "closed": self.closed,
      "centerline": np.round(self.centerline, 4).tolist(),
      "width_left": np.round(self.width_left, 4).tolist(),
      "width_right": np.round(self.width_right, 4).tolist(),
      "cones": [{**asdict(c), "color": str(c.color)} for c in self.cones],
      "start_pose": list(self.start_pose),
      "timekeeping_line": self.timekeeping_line,
      "finish_line": self.finish_line,
      "stop_zone": self.stop_zone,
      "route": self.route,
      "laps": self.laps,
      "stats": asdict(self.stats),
      "generator": self.generator,
      "schema_version": self.schema_version,
    }
    return d

  @classmethod
  def from_dict(cls, d: dict) -> Track:
    def line(v):
      return None if v is None else (tuple(v[0]), tuple(v[1]))

    return cls(
      id=d["id"],
      discipline=d["discipline"],
      seed=d["seed"],
      closed=d["closed"],
      centerline=np.asarray(d["centerline"], dtype=np.float64),
      width_left=np.asarray(d["width_left"], dtype=np.float64),
      width_right=np.asarray(d["width_right"], dtype=np.float64),
      cones=[Cone(c["x"], c["y"], ConeColor(c["color"]), c.get("role", "boundary"), c.get("id", -1))
             for c in d["cones"]],
      start_pose=tuple(d["start_pose"]),
      timekeeping_line=line(d.get("timekeeping_line")),
      finish_line=line(d.get("finish_line")),
      stop_zone=[tuple(p) for p in d["stop_zone"]] if d.get("stop_zone") else None,
      route=d.get("route", []),
      laps=d.get("laps", 1),
      stats=TrackStats(**d.get("stats", {})),
      generator=d.get("generator", {}),
      schema_version=d.get("schema_version", SCHEMA_VERSION),
    )
