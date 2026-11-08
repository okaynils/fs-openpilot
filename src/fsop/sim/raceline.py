"""Offline racing reference: bounded lateral offsets and minimum travel time.

The centerline is a route coordinate, not the desired driving path. Optimization
is local (not a claim of global optimality); all candidates share its topology.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from fsop.sim.vehicle import VehicleParams
from fsop.track import centerline as cl
from fsop.track.schema import Track

PLANNER_VERSION = 6


@dataclass
class RacingLimits:
  v_max: float = 14.0
  a_lat: float = 8.0
  a_acc: float = 5.0
  a_brake: float = 8.0
  clearance: float = 0.60  # desired body reserve; reduced clearance lowers planned speed
  clearance_min: float = 0.35  # hard body margin, including the cone base
  knot_spacing: float = 8.0
  max_iterations: int = 60


@dataclass
class RacingLine:
  points: np.ndarray
  speed: np.ndarray
  offset: np.ndarray
  travel_time: float
  baseline_time: float
  baseline_feasible: bool
  optimized: bool
  solver_message: str
  converged: bool
  iterations: int
  clearance_required: float


def velocity_profile(points: np.ndarray, closed: bool, limits: RacingLimits,
                     vehicle: VehicleParams, clearance: np.ndarray | None = None) -> np.ndarray:
  """Speed envelope with actual segment lengths and a combined grip ellipse.

  Both ends of each segment must admit its longitudinal acceleration. Iterative
  relaxation handles braking across the periodic seam as well as open routes.
  This is a flying speed envelope; mission start/stop constraints are separate.
  """
  kappa = np.abs(cl.curvature(points, closed))
  lateral = min(limits.a_lat, 0.85 * vehicle.mu * 9.81)
  acc = min(limits.a_acc, vehicle.max_accel, 0.8 * vehicle.mu * 9.81)
  brake = min(limits.a_brake, vehicle.max_brake, vehicle.mu * 9.81)
  if min(lateral, acc, brake, limits.v_max) <= 0:
    raise ValueError("Racing speed and acceleration limits must be positive")
  n = len(points)
  a = np.arange(n if closed else n - 1)
  b = (a + 1) % n
  ds = np.linalg.norm(points[b] - points[a], axis=1)
  if np.any(ds < 1e-6):
    raise ValueError("Racing path contains duplicate consecutive points")
  # Leave longitudinal grip at the corner-speed limit. Using the physical grip
  # limit here (rather than a_lat) also supports conservative expert settings.
  grip = vehicle.mu * 9.81
  w = np.minimum(limits.v_max**2, lateral / np.maximum(kappa, 1e-9))
  if clearance is not None:
    # Tracking reserve is a planning budget: speed scales with available reserve,
    # and lateral acceleration with its square. Brake propagation below makes
    # the reduction anticipatory, rather than slowing only at the bottleneck.
    w *= np.clip(np.asarray(clearance) / limits.clearance, 0.05, 1.0)**2
  steer = np.arctan(vehicle.wheelbase * cl.curvature(points, closed))
  rate = np.abs(steer[b] - steer[a]) / ds
  rate_v = vehicle.max_steer_rate / np.maximum(rate, 1e-9)
  np.minimum.at(w, a, rate_v**2)
  np.minimum.at(w, b, rate_v**2)
  for _ in range(2 * n):
    previous = w.copy()
    fraction = np.sqrt(np.maximum(0.0, 1.0 - (w * kappa / grip)**2))
    available = np.minimum(fraction[a], fraction[b])
    # Include aerodynamic drag in the acceleration budget.
    drive = np.maximum(0.0, acc * available - vehicle.drag * np.maximum(w[a], w[b]) / vehicle.mass)
    w[b] = np.minimum(w[b], w[a] + 2 * drive * ds)
    w[a] = np.minimum(w[a], w[b] + 2 * brake * available * ds)
    if np.max(previous - w) < 1e-8:
      break
  return np.sqrt(w)


def travel_time(points: np.ndarray, speed: np.ndarray, closed: bool) -> float:
  a = np.arange(len(points) if closed else len(points) - 1)
  b = (a + 1) % len(points)
  return float(np.sum(2 * np.linalg.norm(points[b] - points[a], axis=1) /
                      np.maximum(speed[a] + speed[b], 1e-6)))


def optimize_raceline(track: Track, vehicle: VehicleParams, limits: RacingLimits | None = None,
                      cones_xy: np.ndarray | None = None) -> RacingLine:
  """Optimize smooth offsets for travel time inside the available track corridor.

  ``cones_xy`` overrides nominal cones with the actual episode geometry.
  Endpoints of open routes remain on their original line; closed routes use a
  periodic spline. The dense track sampling is used in both costs and constraints
  so spline overshoot cannot silently escape the corridor. Every retained iterate
  must satisfy all constraints and improve on a feasible baseline. If the baseline
  is infeasible, a feasible detour may take longer. Budget-limited progress is
  retained with converged=False; an unchecked iterate is never used.
  """
  limits = limits or RacingLimits()
  center = np.asarray(track.centerline, dtype=float)
  if len(center) < 4 or not np.isfinite(center).all():
    raise ValueError("Racing reference requires at least four finite route points")
  from fsop.track.rules import RULES

  cone_radius = max(RULES.small_cone.base, RULES.large_cone.base) / 2
  if not cone_radius < limits.clearance_min <= limits.clearance or limits.knot_spacing <= 0:
    raise ValueError("Clearances must exceed cone radius, desired >= minimum, and knot spacing must be positive")
  normal = cl.normals(center, track.closed)
  station = track.s
  length = track.length
  count = max(4, int(np.ceil(length / limits.knot_spacing)))
  knots = np.linspace(0, length, count + 1)
  # Linear map from spline controls to dense offsets, computed once.
  if track.closed:
    basis = np.vstack([np.eye(count), np.eye(count)[0]])
    matrix = CubicSpline(knots, basis, bc_type="periodic")(station)
  else:
    basis = np.eye(count + 1)[:, 1:-1]
    matrix = CubicSpline(knots, basis, bc_type="clamped")(station)
  half = vehicle.width / 2
  extent = max(vehicle.lf + vehicle.front_overhang, vehicle.lr + vehicle.rear_overhang)
  lower = -np.asarray(track.width_right) + half
  upper = np.asarray(track.width_left) - half
  if np.any(lower + limits.clearance_min > 0) or np.any(upper - limits.clearance_min < 0):
    raise ValueError("Track is too narrow for the vehicle and requested clearance")
  k_max = np.tan(vehicle.max_steer) / vehicle.wheelbase
  cone_xy = track.cone_array()[0] if cones_xy is None else np.asarray(cones_xy, dtype=float)
  if cone_xy.ndim != 2 or cone_xy.shape[1] != 2 or not np.isfinite(cone_xy).all():
    raise ValueError("Scene cones must be a finite [N,2] array")
  cone_tree = cKDTree(cone_xy) if len(cone_xy) else None

  def cone_clearance(points):
    if cone_tree is None:
      return np.full(len(points), limits.clearance + 1.0)
    # Check the oriented body, not just the CoG. Query a radius covering the
    # complete footprint plus reserve; other cones cannot intersect it.
    pairs = cone_tree.query_ball_point(points, extent + half + limits.clearance)
    rows = np.repeat(np.arange(len(points)), [len(ids) for ids in pairs])
    if not len(rows):
      return np.full(len(points), limits.clearance + 1.0)
    ids = np.concatenate([np.asarray(ids, dtype=int) for ids in pairs])
    delta = cone_xy[ids] - points[rows]
    h = cl.headings(points, track.closed)[rows]
    x = np.cos(h) * delta[:, 0] + np.sin(h) * delta[:, 1]
    y = -np.sin(h) * delta[:, 0] + np.cos(h) * delta[:, 1]
    dx = np.maximum(np.maximum(-(vehicle.lr + vehicle.rear_overhang) - x,
                               x - (vehicle.lf + vehicle.front_overhang)), 0)
    dy = np.maximum(np.abs(y) - half, 0)
    clearance = np.full(len(points), limits.clearance + 1.0)
    np.minimum.at(clearance, rows, np.hypot(dx, dy))
    return clearance

  def geometry(x):
    offset = matrix @ x
    points = center + normal * offset[:, None]
    return offset, points

  def path_clearance(x):
    offset, points = geometry(x)
    heading_error = cl.headings(points, track.closed) - cl.headings(center, track.closed)
    # Heading change consumes corridor width with the nose/tail.
    extra = extent * np.abs(np.sin(heading_error))
    margin = np.minimum(np.minimum(offset - lower - extra, upper - offset - extra), cone_clearance(points))
    return points, margin

  required_clearance = limits.clearance

  def constraints(x):
    points, margin = path_clearance(x)
    return np.r_[margin - required_clearance, k_max - np.abs(cl.curvature(points, track.closed))]

  def profile(x):
    points, margin = path_clearance(x)
    return velocity_profile(points, track.closed, limits, vehicle, clearance=margin)

  def objective(x):
    _, points = geometry(x)
    return travel_time(points, profile(x), track.closed)

  zero = np.zeros(matrix.shape[1])
  base_speed = profile(zero)
  base_time = travel_time(center, base_speed, track.closed)
  def solve(initial):
    baseline_feasible = bool(np.min(constraints(zero)) >= -1e-6)
    best_x = zero if baseline_feasible else None
    best_time = base_time if baseline_feasible else float("inf")

    def consider(x):
      nonlocal best_x, best_time
      if not np.isfinite(x).all() or np.min(constraints(x)) < -1e-6:
        return
      time = objective(x)
      if np.isfinite(time) and time < best_time - 1e-5:
        best_x, best_time = x.copy(), time

    consider(initial)
    result = minimize(objective, initial, method="SLSQP", constraints={"type": "ineq", "fun": constraints},
                      callback=consider, options={"maxiter": limits.max_iterations, "ftol": 1e-5})
    consider(result.x)
    return result, best_x, best_time, baseline_feasible

  result, best_x, best_time, baseline_feasible = solve(zero)
  iterations = int(result.nit)
  message = str(result.message)
  if best_x is None and limits.clearance_min < limits.clearance:
    # Prioritize the desired reserve over lap time. Only use the reduced hard
    # margin when the first solve cannot produce any feasible reference; the
    # speed envelope then pays for reduced reserve before reaching the bottleneck.
    required_clearance = limits.clearance_min
    initial = result.x if np.isfinite(result.x).all() and np.min(constraints(result.x)) >= 0 else zero
    result, best_x, best_time, baseline_feasible = solve(initial)
    iterations += int(result.nit)
    message = "Desired reserve unavailable within solver budget; " + str(result.message)
  if best_x is None:
    raise ValueError(f"No feasible racing reference: {message}")
  accepted = not baseline_feasible or best_time < base_time - 1e-5
  offset, points = geometry(best_x)
  speed = profile(best_x) if accepted else base_speed
  return RacingLine(points, speed, offset, travel_time(points, speed, track.closed), base_time, baseline_feasible,
                    accepted, message, bool(result.success), iterations, required_clearance)
