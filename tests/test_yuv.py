import numpy as np

from fsop.geometry.yuv import model_to_rgb, pack_yuv6, rgb_to_model, rgb_to_nv12, rgb_to_yuv420


def _reference_nv12(rgb):
  """Verbatim copy of openpilot tools/sim/lib/camerad.py::rgb_to_nv12 (MIT)."""
  h, w = rgb.shape[:2]
  r = rgb[:, :, 0].astype(np.int32)
  g = rgb[:, :, 1].astype(np.int32)
  b = rgb[:, :, 2].astype(np.int32)
  y = (((b * 13 + g * 65 + r * 33) + 64) >> 7) + 16
  y = np.clip(y, 0, 255).astype(np.uint8)
  r_sub = (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2] + 2) >> 2
  g_sub = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2
  b_sub = (b[0::2, 0::2] + b[0::2, 1::2] + b[1::2, 0::2] + b[1::2, 1::2] + 2) >> 2
  u = np.clip((b_sub * 56 - g_sub * 37 - r_sub * 19 + 0x8080) >> 8, 0, 255).astype(np.uint8)
  v = np.clip((r_sub * 56 - g_sub * 47 - b_sub * 9 + 0x8080) >> 8, 0, 255).astype(np.uint8)
  uv = np.empty((h // 2, w), dtype=np.uint8)
  uv[:, 0::2] = u
  uv[:, 1::2] = v
  return np.concatenate([y.ravel(), uv.ravel()]).tobytes()


def test_nv12_bit_exact():
  rgb = np.random.default_rng(0).integers(0, 256, (256, 512, 3), dtype=np.uint8)
  assert rgb_to_nv12(rgb) == _reference_nv12(rgb)


def test_pack_layout_matches_frames_to_tensor():
  rgb = np.random.default_rng(1).integers(0, 256, (256, 512, 3), dtype=np.uint8)
  y, u, v = rgb_to_yuv420(rgb)
  x = pack_yuv6(y, u, v)
  H = 256
  # frames_to_tensor on the stacked [Y; U; V] buffer of shape (H*3/2, W)
  frames = np.concatenate([y, u.reshape(-1, 512), v.reshape(-1, 512)], 0)
  ref = np.stack([frames[0:H:2, 0::2], frames[1:H:2, 0::2], frames[0:H:2, 1::2], frames[1:H:2, 1::2],
                  frames[H:H + H // 4].reshape(H // 2, 256), frames[H + H // 4:H + H // 2].reshape(H // 2, 256)])
  assert x.shape == (6, 128, 256)
  np.testing.assert_array_equal(x, ref)


def test_roundtrip_close():
  rgb = np.full((256, 512, 3), (60, 120, 200), np.uint8)
  back = model_to_rgb(rgb_to_model(rgb))
  assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 4
