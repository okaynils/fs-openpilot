"""Procedural rule-compliant trackdrive/autocross generation.

Two centerline generators:
  * ``fourier``: radial Fourier perturbation of a circle -> organic flowing loops.
  * ``curvature``: random corner/straight sequence integrated in the curvature domain and closed by
    least squares -> exact control of minimum radius and straight length; hairpins, chicanes.
Both are followed by optional slalom injection, a smooth width profile, cone placement and the
rule validator; generation rejection-samples until a track passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from fsop.track import centerline as cl
from fsop.track.cones import add_start_gate, place_boundary_cones
from fsop.track.fixed_layouts import REF_TO_FRONT
from fsop.track.rules import RULES
from fsop.track.schema import Track, TrackStats

DS = 0.25


@dataclass
class TrackGenCfg:
  discipline: str = "trackdrive"  # trackdrive | autocross
  method: str = "mixed"  # fourier | curvature | mixed
  length_min: float = 220.0
  length_max: float = 480.0
  width_min: float = 3.0
  width_max: float = 4.5
  min_center_radius: float = 4.75  # >= 4.5 m (9 m turning diameter) + margin
  slalom_prob: float = 0.35
  slalom_spacing: tuple[float, float] = (7.5, 12.0)
  slalom_amplitude: tuple[float, float] = (1.0, 1.8)
  max_tries: int = 200
  cone_spacing_min: float = 1.5
  cone_spacing_max: float = 5.0
  cone_spacing_k: float = 0.35  # spacing = k * boundary radius, clipped
  extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# centerline generators (return closed polyline, roughly correct scale)
# ---------------------------------------------------------------------------------------------

def _fourier_loop(rng: np.random.Generator, n: int = 720) -> np.ndarray:
  th = np.linspace(0, 2 * np.pi, n, endpoint=False)
  r = np.ones_like(th)
  amp = rng.uniform(0.25, 0.55)
  for k in range(2, 7):
    r += amp * rng.uniform(-1, 1) / k**1.3 * np.cos(k * th + rng.uniform(0, 2 * np.pi))
  r = np.maximum(r, 0.25)
  pts = np.stack([r * np.cos(th), r * np.sin(th)], 1)
  pts[:, 0] *= rng.uniform(1.0, 2.2)  # anisotropy
  return pts


def _curvature_loop(rng: np.random.Generator, length: float, r_min: float, wander: float = 2.0) -> np.ndarray:
  """Random corner/straight sequence in the curvature domain, steered so the net heading is 2*pi,
  then closed exactly in heading (scaling left/right turns) and approximately in position
  (low-frequency curvature harmonics + linear spread of the small residual)."""
  segs: list[tuple[float, float]] = []
  s = heading = 0.0
  while s < length:
    if rng.random() < 0.25:
      seg = rng.uniform(4.0, 40.0)
      segs.append((seg, 0.0))
    else:
      R = r_min + rng.gamma(1.2, 3.5)
      ang = np.deg2rad(rng.uniform(25.0, 200.0))
      target = 2 * np.pi * s / length
      p_left = 1 / (1 + np.exp(-(target - heading + 0.3) / wander))
      sign = 1.0 if rng.random() < p_left else -1.0
      seg = R * ang
      segs.append((seg, sign / R))
      heading += sign * ang
    s += seg
  k = np.concatenate([np.full(max(int(seg / DS), 1), kk) for seg, kk in segs])
  k = gaussian_filter1d(k, 1.5 / DS, mode="wrap")  # clothoid-like transitions

  n = len(k)
  u = np.arange(n) / n
  pos, neg = np.clip(k, 0, None), np.clip(k, None, 0)
  basis = np.array([np.cos(2 * np.pi * u), np.sin(2 * np.pi * u), np.cos(4 * np.pi * u), np.sin(4 * np.pi * u)])
  L = n * DS

  def make(q):
    return pos * np.exp(q[0]) + neg * np.exp(q[1]) + q[2:] @ basis

  def integrate(kk):
    h = np.concatenate([[0.0], np.cumsum(kk)[:-1] * DS])
    return np.stack([np.cumsum(np.cos(h)) * DS, np.cumsum(np.sin(h)) * DS], 1), float(np.sum(kk) * DS)

  def residual(q):
    xy, H = integrate(make(q))
    return np.r_[(H - 2 * np.pi) * L, xy[-1], 2 * L * q[2:], 5.0 * q[:2]]

  q = least_squares(residual, np.zeros(6)).x
  xy, _ = integrate(make(q))
  err = xy[-1] - xy[0] + (xy[1] - xy[0])
  return xy - np.outer(u, err)


def _scale_to_length(p: np.ndarray, target: float) -> np.ndarray:
  p = cl.resample(p, ds=cl.total_length(p, True) / 1000, closed=True)
  L = cl.total_length(p, True)
  p = (p - p.mean(0)) * (target / L)
  return p


def _smooth_curvature(p: np.ndarray, sigma_m: float = 2.0) -> np.ndarray:
  sig = sigma_m / DS
  return np.stack([gaussian_filter1d(p[:, 0], sig, mode="wrap"), gaussian_filter1d(p[:, 1], sig, mode="wrap")], 1)


def _inject_slalom(p: np.ndarray, rng: np.random.Generator, cfg: TrackGenCfg) -> tuple[np.ndarray, dict | None]:
  """Replace the longest straight's middle with a sinusoidal slalom."""
  k = cl.curvature(p, True)
  straight = np.abs(k) < 1 / 60
  if not straight.any():
    return p, None
  start = int(np.argmin(straight))
  rolled = np.roll(straight, -start)
  best_len, best_i, cur, cur_i = 0, 0, 0, 0
  for i, m in enumerate(rolled):
    if m:
      if cur == 0:
        cur_i = i
      cur += 1
      if cur > best_len:
        best_len, best_i = cur, cur_i
    else:
      cur = 0
  L = best_len * DS
  spacing = rng.uniform(*cfg.slalom_spacing)
  n_half = int((L - 10.0) // spacing)
  if n_half < 3:
    return p, None
  n_half = min(n_half, 6)
  amp = rng.uniform(*cfg.slalom_amplitude)
  seg_len = n_half * spacing
  i0 = (start + best_i + int((L - seg_len) / 2 / DS)) % len(p)
  m = int(seg_len / DS)
  idx = (i0 + np.arange(m)) % len(p)
  u = np.arange(m) * DS / seg_len
  offs = amp * np.sin(np.pi * n_half * u)
  # taper ends to keep C1 continuity
  taper = np.clip(np.minimum(u, 1 - u) / 0.08, 0, 1)
  offs *= taper
  nrm = cl.normals(p, True)[idx]
  p = p.copy()
  p[idx] += nrm * offs[:, None]
  return p, {"spacing": spacing, "amplitude": amp, "n": n_half, "s0": float(i0 * DS)}


def _width_profile(n: int, rng: np.random.Generator, cfg: TrackGenCfg) -> tuple[np.ndarray, np.ndarray]:
  base = rng.uniform(cfg.width_min + 0.2, cfg.width_max)
  noise = gaussian_filter1d(rng.normal(0, 1, n), sigma=60 / DS, mode="wrap")
  noise /= max(np.abs(noise).max(), 1e-9)
  w = np.clip(base + noise * rng.uniform(0, 0.8), cfg.width_min, cfg.width_max)
  return w / 2, w / 2


# ---------------------------------------------------------------------------------------------

def generate_track(seed: int, cfg: TrackGenCfg | None = None) -> Track:
  """Generate one rule-compliant closed-loop track (rejection sampling on the validator)."""
  from fsop.track.validate import validate

  cfg = cfg or TrackGenCfg()
  rng = np.random.default_rng(seed)
  last_report = None
  for attempt in range(cfg.max_tries):
    method = cfg.method if cfg.method != "mixed" else ("curvature" if rng.random() < 0.85 else "fourier")
    target = rng.uniform(cfg.length_min, cfg.length_max)
    if method == "curvature":
      p = cl.resample(_curvature_loop(rng, target, cfg.min_center_radius + 0.25), DS, closed=True)
    else:
      p = _scale_to_length(_fourier_loop(rng), target)
      p = cl.resample(p, DS, closed=True)
      p = _smooth_curvature(p, sigma_m=rng.uniform(1.0, 3.0))
    slalom = None
    if rng.random() < cfg.slalom_prob:
      p, slalom = _inject_slalom(p, rng, cfg)
    p = cl.resample(p, DS, closed=True)
    k = cl.curvature(p, True)
    if np.max(np.abs(k)) > 1 / cfg.min_center_radius:
      continue
    # ensure counter-clockwise traversal is not required: randomize direction
    if rng.random() < 0.5:
      p = p[::-1].copy()
    wl, wr = _width_profile(len(p), rng, cfg)
    # start at a point on a straight-ish segment
    k = cl.curvature(p, True)
    straightness = gaussian_filter1d(np.abs(k), 20 / DS, mode="wrap")
    i_start = int(np.argmin(straightness))
    p = np.roll(p, -i_start, axis=0)
    wl, wr = np.roll(wl, -i_start), np.roll(wr, -i_start)
    track = Track(
      id=f"{cfg.discipline}_{seed:06d}",
      discipline=cfg.discipline,  # type: ignore[arg-type]
      seed=seed,
      closed=True,
      centerline=p,
      width_left=wl,
      width_right=wr,
      laps=RULES.trackdrive_laps if cfg.discipline == "trackdrive" else 1,
      generator={"method": method, "attempt": attempt, "target_length": target, "slalom": slalom},
    )
    finalize_track(track, cfg)
    report = validate(track)
    last_report = report
    if report.passed:
      return track
  raise RuntimeError(f"no valid track after {cfg.max_tries} tries (seed {seed}): {last_report}")


def finalize_track(track: Track, cfg: TrackGenCfg | None = None) -> Track:
  """Place cones, start gate, staging pose and stop zone for a closed loop."""
  cfg = cfg or TrackGenCfg()
  left, right = track.boundaries()
  track.cones = (place_boundary_cones(left, "left", track.closed, cfg)
                 + place_boundary_cones(right, "right", track.closed, cfg))
  add_start_gate(track, s_line=0.0)
  h = track.heading()
  staging = RULES.trackdrive_staging.value if track.discipline == "trackdrive" else RULES.autocross_staging.value
  i_stage = int(round(-(staging + REF_TO_FRONT) / DS)) % len(track.centerline)
  x, y = track.centerline[i_stage]
  track.start_pose = (float(x), float(y), float(h[i_stage]))
  # stop zone: 30 m after the line, as polygon between boundaries
  stop_len = RULES.trackdrive_stop_zone.value
  n_stop = int(stop_len / DS)
  track.stop_zone = [tuple(map(float, q)) for q in np.vstack([left[:n_stop], right[:n_stop][::-1]])]
  for i, c in enumerate(track.cones):
    c.id = i
  kappa = track.curvature()
  track.stats = TrackStats(
    length=track.length,
    min_radius=float(1 / max(np.abs(kappa).max(), 1e-9)),
    max_straight=cl.straight_runs(kappa, DS, track.closed),
    min_width=float((track.width_left + track.width_right).min()),
    n_cones=len(track.cones),
  )
  return track
