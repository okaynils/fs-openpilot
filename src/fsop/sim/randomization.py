"""Domain randomization: scene-level parameters (per episode) and image post-processing (per frame)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DRCfg:
  level: str = "light"  # off | light | heavy
  cone_jitter: float = 0.05  # [m] position noise applied at sim time
  cone_missing: float = 0.02  # probability a boundary cone is removed
  cone_extra: float = 0.0  # extra orange/unknown cones per 100 m
  cam_height_jitter: float = 0.1  # +/- m around the profile's nominal mount height
  cam_pitch_deg: float = 1.0  # +/- range around the nominal mount angles
  cam_roll_deg: float = 0.5
  cam_yaw_deg: float = 0.5
  image_noise: float = 2.0  # max gaussian sigma (in 0..255)
  blur_prob: float = 0.1
  jpeg_prob: float = 0.2
  exposure: tuple[float, float] = (0.85, 1.15)
  # venue realism (sky panoramas, event-ground asphalt, perimeter scenery, people, shadows, camera
  # degradation): probability per episode; 0 keeps the classic scenes bit-for-bit
  venue_prob: float = 0.0
  panorama_dir: str = "artifacts/panoramas"


def dr_preset(level: str) -> DRCfg:
  if level == "off":
    return DRCfg("off", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (1.0, 1.0))
  if level == "heavy":
    return DRCfg("heavy", 0.12, 0.08, 2.0, 0.25, 3.0, 1.5, 1.5, 6.0, 0.3, 0.5, (0.6, 1.4))
  return DRCfg()


@dataclass
class SceneDR:
  sun_azimuth: float = 0.8
  sun_elevation: float = 0.9
  sun_intensity: float = 1.0
  sun_color: tuple[float, float, float] = (1.0, 0.97, 0.92)
  ambient: float = 0.45
  fog_density: float = 0.0
  sky_color: tuple[float, float, float] = (0.62, 0.74, 0.90)
  ground_kind: str = "procedural"  # procedural | metadrive
  ground_base: float = 0.33
  ground_contrast: float = 0.06
  ground_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
  grass_ring: bool = True
  cone_brightness_sigma: float = 0.05
  boundary_lines: bool = False  # chalk/paint lines along the track boundaries (common at FS events)
  line_color: tuple[float, float, float] = (0.85, 0.85, 0.8)
  line_width: float = 0.08
  line_gap_prob: float = 0.1
  cam_dz: float = 0.0  # mount perturbations relative to the team profile's nominal camera mount
  cam_pitch: float = 0.0
  cam_roll: float = 0.0
  cam_yaw: float = 0.0
  exposure: float = 1.0
  noise_sigma: float = 1.0
  blur: bool = False
  jpeg_quality: int = 0  # 0 = off
  seed: int = 0
  # venue realism (see fsop.sim.render.venue); neutral defaults = classic scene
  venue: bool = False
  sky_pano: str = ""  # equirectangular photo; "" = flat sky colour
  sky_yaw: float = 0.0
  shadow_alpha: float = 0.0
  cone_fade: float = 0.0  # mix cone colours towards pale / sun-bleached
  cone_lettering: bool = False
  haze: float = 0.0  # veil towards a bright grey
  saturation: float = 1.0
  white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
  resample: float = 1.0  # <1: render detail lost (down- then up-sampled)
  vignette: float = 0.0
  extra: dict = field(default_factory=dict)

  def to_dict(self) -> dict:
    return asdict(self)


def sample_scene(rng: np.random.Generator, cfg: DRCfg) -> SceneDR:
  if cfg.level == "off":
    return SceneDR(seed=int(rng.integers(1 << 30)), noise_sigma=0.0)
  heavy = cfg.level == "heavy"
  warm = rng.uniform(0.0, 0.15 if not heavy else 0.3)
  scene = SceneDR(
    sun_azimuth=float(rng.uniform(0, 2 * np.pi)),
    sun_elevation=float(rng.uniform(0.25 if heavy else 0.4, 1.3)),
    sun_intensity=float(rng.uniform(0.6, 1.3) if heavy else rng.uniform(0.8, 1.15)),
    sun_color=(1.0, float(1.0 - warm * 0.4), float(1.0 - warm)),
    ambient=float(rng.uniform(0.25, 0.65)),
    fog_density=float(rng.uniform(0, 0.012) if heavy and rng.random() < 0.4 else 0.0),
    sky_color=tuple(float(v) for v in np.clip(np.array([0.62, 0.74, 0.90]) + rng.normal(0, 0.08, 3), 0, 1)),
    ground_kind="metadrive" if rng.random() < 0.3 else "procedural",
    ground_base=float(rng.uniform(0.22, 0.45)),
    ground_contrast=float(rng.uniform(0.03, 0.12)),
    ground_tint=tuple(float(v) for v in 1 + rng.normal(0, 0.04, 3)),
    grass_ring=bool(rng.random() < 0.7),
    cone_brightness_sigma=0.1 if heavy else 0.05,
    boundary_lines=bool(rng.random() < 0.5),
    line_color=tuple(float(v) for v in np.array((0.85, 0.85, 0.8) if rng.random() < 0.75 else (0.35, 0.8, 0.35))
                     * rng.uniform(0.75, 1.05)),
    line_width=float(rng.uniform(0.05, 0.12)),
    line_gap_prob=float(rng.uniform(0.0, 0.25)),
    cam_dz=float(rng.uniform(-cfg.cam_height_jitter, cfg.cam_height_jitter)),
    cam_pitch=float(np.deg2rad(rng.uniform(-cfg.cam_pitch_deg, cfg.cam_pitch_deg))),
    cam_roll=float(np.deg2rad(rng.uniform(-cfg.cam_roll_deg, cfg.cam_roll_deg))),
    cam_yaw=float(np.deg2rad(rng.uniform(-cfg.cam_yaw_deg, cfg.cam_yaw_deg))),
    exposure=float(rng.uniform(*cfg.exposure)),
    noise_sigma=float(rng.uniform(0, cfg.image_noise)),
    blur=bool(rng.random() < cfg.blur_prob),
    jpeg_quality=int(rng.integers(35, 90)) if rng.random() < cfg.jpeg_prob else 0,
    seed=int(rng.integers(1 << 30)),
  )
  if cfg.venue_prob > 0 and rng.random() < cfg.venue_prob:  # draws only when enabled: classic sets reproduce
    _venue(rng, cfg, scene)
  return scene


def _venue(rng: np.random.Generator, cfg: DRCfg, sc: SceneDR) -> None:
  from fsop.sim.render.venue import available_panoramas

  panos = available_panoramas(cfg.panorama_dir)
  sc.venue = True
  sc.sky_pano = str(Path(cfg.panorama_dir) / f"{panos[int(rng.integers(len(panos)))]}.jpg") \
    if panos and rng.random() < 0.9 else ""
  sc.sky_yaw = float(rng.uniform(0, 2 * np.pi))
  sc.ground_kind = "venue"
  sc.ground_base = float(rng.uniform(0.36, 0.62))  # event asphalt is often light and sun-bleached
  warm = rng.uniform(0.0, 0.06)  # event asphalt: neutral to warm grey, sometimes slightly green
  sc.ground_tint = tuple(float(v) for v in np.array([1 + warm, 1 + 0.8 * warm + rng.uniform(0, 0.03), 1 - 0.6 * warm])
                         + rng.normal(0, 0.012, 3))
  sc.grass_ring = bool(rng.random() < 0.25)
  sc.line_color = tuple(float(v) for v in np.array([(0.85, 0.85, 0.8), (0.35, 0.8, 0.35), (0.85, 0.8, 0.3)][
    int(rng.choice(3, p=[0.45, 0.4, 0.15]))]) * rng.uniform(0.75, 1.05))  # spray lines: green is common
  sc.shadow_alpha = float(np.clip(rng.uniform(0.3, 0.7) * sc.sun_intensity / (sc.sun_intensity + sc.ambient) * 1.5,
                                  0.1, 0.75))
  sc.cone_fade = float(rng.uniform(0.0, 0.5))
  sc.cone_lettering = bool(rng.random() < 0.6)
  sc.haze = float(rng.uniform(0.0, 0.25))
  sc.saturation = float(rng.uniform(0.6, 1.05))
  sc.white_balance = tuple(float(v) for v in 1 + rng.normal(0, 0.04, 3))
  sc.resample = float(rng.uniform(0.6, 1.0)) if rng.random() < 0.6 else 1.0
  sc.vignette = float(rng.uniform(0.0, 0.35))
  sc.blur = bool(rng.random() < 0.3)
  sc.jpeg_quality = int(rng.integers(45, 90)) if rng.random() < 0.4 else 0


def postprocess(img: np.ndarray, dr: SceneDR, rng: np.random.Generator, yaw_rate: float = 0.0) -> np.ndarray:
  """Photometric randomization on a rendered RGB uint8 frame."""
  x = img.astype(np.float32)
  if dr.exposure != 1.0:
    x = 255.0 * np.power(np.clip(x / 255.0 * dr.exposure, 0, 1), 1.0 / max(dr.exposure, 0.5) ** 0.3)
  if dr.blur or abs(yaw_rate) > 1.0:
    k = 3 if abs(yaw_rate) < 2.0 else 5
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    x = cv2.filter2D(x, -1, kernel)
  if dr.noise_sigma > 0:
    x = x + rng.normal(0, dr.noise_sigma, x.shape)
  if dr.venue:
    x = _camera(x, dr)
  out = np.clip(x, 0, 255).astype(np.uint8)
  if dr.jpeg_quality:
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, dr.jpeg_quality])
    if ok:
      out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
  return out


def _camera(x: np.ndarray, dr: SceneDR) -> np.ndarray:
  """Venue camera look: white balance, saturation, haze veil, lost detail, vignetting (float 0..255)."""
  x = x * np.asarray(dr.white_balance, np.float32)
  if dr.saturation != 1.0:
    g = x.mean(-1, keepdims=True)
    x = g + dr.saturation * (x - g)
  if dr.haze > 0:
    x = x * (1 - dr.haze) + 200.0 * dr.haze
  if dr.resample < 1.0:
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(int(w * dr.resample), 8), max(int(h * dr.resample), 8)), interpolation=cv2.INTER_AREA)
    x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
  if dr.vignette > 0:
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    x = x * (1 - dr.vignette * np.clip(r2 / 2, 0, 1))[..., None]
  return x
