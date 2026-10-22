"""Domain randomization: scene-level parameters (per episode) and image post-processing (per frame)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np


@dataclass
class DRCfg:
  level: str = "light"  # off | light | heavy
  cone_jitter: float = 0.05  # [m] position noise applied at sim time
  cone_missing: float = 0.02  # probability a boundary cone is removed
  cone_extra: float = 0.0  # extra orange/unknown cones per 100 m
  cam_height: tuple[float, float] = (1.0, 1.2)
  cam_pitch_deg: float = 1.0  # +/- range
  cam_roll_deg: float = 0.5
  cam_yaw_deg: float = 0.5
  image_noise: float = 2.0  # max gaussian sigma (in 0..255)
  blur_prob: float = 0.1
  jpeg_prob: float = 0.2
  exposure: tuple[float, float] = (0.85, 1.15)


def dr_preset(level: str) -> DRCfg:
  if level == "off":
    return DRCfg("off", 0.0, 0.0, 0.0, (1.1, 1.1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (1.0, 1.0))
  if level == "heavy":
    return DRCfg("heavy", 0.12, 0.08, 2.0, (0.8, 1.3), 3.0, 1.5, 1.5, 6.0, 0.3, 0.5, (0.6, 1.4))
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
  cam_height: float = 1.1
  cam_pitch: float = 0.0
  cam_roll: float = 0.0
  cam_yaw: float = 0.0
  exposure: float = 1.0
  noise_sigma: float = 1.0
  blur: bool = False
  jpeg_quality: int = 0  # 0 = off
  seed: int = 0
  extra: dict = field(default_factory=dict)

  def to_dict(self) -> dict:
    return asdict(self)


def sample_scene(rng: np.random.Generator, cfg: DRCfg) -> SceneDR:
  if cfg.level == "off":
    return SceneDR(seed=int(rng.integers(1 << 30)), cam_height=float(cfg.cam_height[0]), noise_sigma=0.0)
  heavy = cfg.level == "heavy"
  warm = rng.uniform(0.0, 0.15 if not heavy else 0.3)
  return SceneDR(
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
    cam_height=float(rng.uniform(*cfg.cam_height)),
    cam_pitch=float(np.deg2rad(rng.uniform(-cfg.cam_pitch_deg, cfg.cam_pitch_deg))),
    cam_roll=float(np.deg2rad(rng.uniform(-cfg.cam_roll_deg, cfg.cam_roll_deg))),
    cam_yaw=float(np.deg2rad(rng.uniform(-cfg.cam_yaw_deg, cfg.cam_yaw_deg))),
    exposure=float(rng.uniform(*cfg.exposure)),
    noise_sigma=float(rng.uniform(0, cfg.image_noise)),
    blur=bool(rng.random() < cfg.blur_prob),
    jpeg_quality=int(rng.integers(35, 90)) if rng.random() < cfg.jpeg_prob else 0,
    seed=int(rng.integers(1 << 30)),
  )


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
  out = np.clip(x, 0, 255).astype(np.uint8)
  if dr.jpeg_quality:
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, dr.jpeg_quality])
    if ok:
      out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
  return out
