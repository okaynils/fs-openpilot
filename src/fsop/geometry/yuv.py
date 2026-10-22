"""RGB <-> openpilot model input (YUV420, 6-channel packing).

``rgb_to_nv12`` is a port of openpilot ``tools/sim/lib/camerad.py`` (MIT): BT.601 integer math with
2x2 box-filtered chroma. ``pack_yuv6`` matches ``frames_to_tensor`` in ``selfdrive/modeld``:
channels = Y[0::2,0::2], Y[1::2,0::2], Y[0::2,1::2], Y[1::2,1::2], U, V  -> [6, H/2, W/2].
"""

from __future__ import annotations

import numpy as np


def rgb_to_yuv420(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """[H,W,3] uint8 RGB -> (Y [H,W], U [H/2,W/2], V [H/2,W/2]) uint8."""
  r = rgb[..., 0].astype(np.int32)
  g = rgb[..., 1].astype(np.int32)
  b = rgb[..., 2].astype(np.int32)
  y = np.clip((((b * 13 + g * 65 + r * 33) + 64) >> 7) + 16, 0, 255).astype(np.uint8)
  r_s = (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2] + 2) >> 2
  g_s = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2
  b_s = (b[0::2, 0::2] + b[0::2, 1::2] + b[1::2, 0::2] + b[1::2, 1::2] + 2) >> 2
  u = np.clip((b_s * 56 - g_s * 37 - r_s * 19 + 0x8080) >> 8, 0, 255).astype(np.uint8)
  v = np.clip((r_s * 56 - g_s * 47 - b_s * 9 + 0x8080) >> 8, 0, 255).astype(np.uint8)
  return y, u, v


def rgb_to_nv12(rgb: np.ndarray) -> bytes:
  """Byte-exact port of openpilot's simulator conversion."""
  h, w = rgb.shape[:2]
  y, u, v = rgb_to_yuv420(rgb)
  uv = np.empty((h // 2, w), dtype=np.uint8)
  uv[:, 0::2] = u
  uv[:, 1::2] = v
  return np.concatenate([y.ravel(), uv.ravel()]).tobytes()


def pack_yuv6(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
  """-> [6, H/2, W/2] uint8 (model input layout for one frame)."""
  return np.stack([y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2], u, v], axis=0)


def rgb_to_model(rgb: np.ndarray) -> np.ndarray:
  """[256,512,3] RGB (already in model frame) -> [6,128,256] uint8."""
  return pack_yuv6(*rgb_to_yuv420(rgb))


def model_to_rgb(x: np.ndarray) -> np.ndarray:
  """[6,128,256] -> [256,512,3] RGB (for visualization; inverse of the conversion above)."""
  c = x.astype(np.float32)
  h2, w2 = c.shape[1:]
  y = np.empty((2 * h2, 2 * w2), np.float32)
  y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2] = c[0], c[1], c[2], c[3]
  # openpilot's chroma coefficients are half of standard BT.601 (0.219 vs 0.439) -> scale back by 2
  u = 2.0 * (np.repeat(np.repeat(c[4], 2, 0), 2, 1) - 128)
  v = 2.0 * (np.repeat(np.repeat(c[5], 2, 0), 2, 1) - 128)
  yy = (y - 16) * 1.164
  r = yy + 1.596 * v
  g = yy - 0.392 * u - 0.813 * v
  b = yy + 2.017 * u
  return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)
