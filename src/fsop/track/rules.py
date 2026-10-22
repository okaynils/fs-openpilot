"""Formula Student Germany 2027 driverless rule constants.

Values come from FS_Rules_2027_v1.0 section D unless marked ``verified=False``; those are
taken from older FS Driverless handbooks / common practice and should be checked against the
current Event Handbook and FS Driverless Specification before relying on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuleValue:
  value: float
  source: str
  verified: bool = True


def _r(value: float, source: str, verified: bool = True) -> RuleValue:
  return RuleValue(value, source, verified)


@dataclass(frozen=True)
class ConeSpec:
  base: float  # square base edge [m]
  height: float  # [m]
  radius_bottom: float  # body radius at the base [m]
  radius_top: float


@dataclass(frozen=True)
class FSGRules2027:
  # Trackdrive / autocross (D 6.1, D 8.1)
  max_straight: RuleValue = _r(80.0, "D 8.1.1")
  min_track_width: RuleValue = _r(3.0, "D 8.1.1")
  # D 1.1.10: diameter of the smallest circle in which the *vehicle* can turn. The track must be
  # drivable by a car with this turning diameter, i.e. centerline radius >= 4.5 m.
  min_turning_diameter: RuleValue = _r(9.0, "D 8.1.1 / D 1.1.10")
  trackdrive_lap_min: RuleValue = _r(200.0, "D 8.1.2")
  trackdrive_lap_max: RuleValue = _r(500.0, "D 8.1.2")
  trackdrive_laps: int = 10  # D 8.3.1
  trackdrive_staging: RuleValue = _r(6.0, "D 8.3.2")
  trackdrive_stop_zone: RuleValue = _r(30.0, "D 8.3.3")
  autocross_max_length: RuleValue = _r(1500.0, "D 6.1.2")
  autocross_slalom_min: RuleValue = _r(7.5, "D 6.1.1")
  autocross_slalom_max: RuleValue = _r(12.0, "D 6.1.1")
  endurance_slalom_min: RuleValue = _r(9.0, "D 7.1.1")
  endurance_slalom_max: RuleValue = _r(15.0, "D 7.1.1")
  autocross_stop_zone: RuleValue = _r(30.0, "D 6.2.4")
  autocross_staging: RuleValue = _r(6.0, "D 6.2.3")
  dq_speed_first3: RuleValue = _r(2.5, "D 2.6.2")
  dq_speed_other: RuleValue = _r(3.5, "D 2.6.2")
  standstill_timeout: RuleValue = _r(30.0, "D 2.6.1")

  # Skidpad (D 4.1)
  skidpad_center_distance: RuleValue = _r(18.25, "D 4.1.2")
  skidpad_inner_diameter: RuleValue = _r(15.25, "D 4.1.2")
  skidpad_outer_diameter: RuleValue = _r(21.25, "D 4.1.2")
  skidpad_inner_cones: int = 17  # D 4.1.3
  skidpad_outer_cones: int = 13  # D 4.1.3
  skidpad_staging: RuleValue = _r(15.0, "D 4.2.4")
  skidpad_stop_zone: RuleValue = _r(25.0, "D 4.2.5")

  # Acceleration (D 5.1)
  accel_length: RuleValue = _r(75.0, "D 5.1.1")
  accel_min_width: RuleValue = _r(3.0, "D 5.1.1")
  accel_cone_spacing: RuleValue = _r(5.0, "D 5.1.1 ('about 5m')")
  accel_staging: RuleValue = _r(0.30, "D 5.2.3")
  accel_stop_zone: RuleValue = _r(75.0, "D 5.2.4")

  # Generator / handbook values (not in the rules PDF)
  max_cone_spacing: RuleValue = _r(5.0, "FS Driverless Handbook (to verify)", verified=False)
  min_cone_spacing_same_side: RuleValue = _r(1.0, "generator choice", verified=False)
  min_section_clearance: RuleValue = _r(2.0, "generator choice", verified=False)

  # Penalties (D 10.1.7): seconds per DOO / OC; None = disqualification.
  doo_penalty_s: dict = field(default_factory=lambda: {
    "acceleration": 2.0, "skidpad": 0.2, "autocross": 2.0, "endurance": 2.0, "trackdrive": 2.0})
  oc_penalty_s: dict = field(default_factory=lambda: {
    "acceleration": None, "skidpad": None, "autocross": 10.0, "endurance": 10.0, "trackdrive": 10.0})

  # Cone geometry (FS Driverless cone spec, to verify)
  small_cone: ConeSpec = field(default_factory=lambda: ConeSpec(0.228, 0.325, 0.095, 0.025))
  large_cone: ConeSpec = field(default_factory=lambda: ConeSpec(0.285, 0.505, 0.125, 0.030))
  cone_dims_verified: bool = False


RULES = FSGRules2027()


def unverified() -> list[tuple[str, RuleValue]]:
  """Return every rule value that still needs checking against the official handbook."""
  out = []
  for name in RULES.__dataclass_fields__:
    v = getattr(RULES, name)
    if isinstance(v, RuleValue) and not v.verified:
      out.append((name, v))
  return out
