"""Spike: run MetaDriveEnv headless on macOS with model-frame cameras; time the render."""
import sys
import time

import numpy as np
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.metadrive_env import MetaDriveEnv
from panda3d.core import GraphicsOutput, Texture


class FSCam(RGBCamera):
  FL, CY = 910.0, 47.6

  def __init__(self, *a, **k):
    super().__init__(*a, **k)
    self.cpu_texture = Texture()
    self.buffer.addRenderTexture(self.cpu_texture, GraphicsOutput.RTMCopyRam)
    lens = self.get_lens()
    lens.setFilmSize(512, 256)
    lens.setFocalLength(self.FL)
    lens.setFilmOffset(0, self.CY - 128)
    lens.setNear(0.1)

  def grab(self):
    t = self.cpu_texture
    return np.frombuffer(t.getRamImageAs("RGB").getData(), np.uint8).reshape(t.getYSize(), t.getXSize(), 3)[::-1]


class FSCamWide(FSCam):
  FL, CY = 455.0, 151.8


cfg = {
  "use_render": False, "image_observation": True, "show_logo": False, "interface_panel": [],
  "vehicle_config": {"image_source": "road", "render_vehicle": False},
  "sensors": {"road": (FSCam, 512, 256), "wide": (FSCamWide, 512, 256)},
  "traffic_density": 0.0, "out_of_route_done": False, "on_continuous_line_done": False,
  "crash_vehicle_done": False, "crash_object_done": False,
  "preload_models": False, "anisotropic_filtering": False, "map": "S", "decision_repeat": 1,
}
env = MetaDriveEnv(cfg)
t0 = time.time()
env.reset()
print("reset s", time.time() - t0)
eng = env.engine
road, wide = eng.get_sensor("road"), eng.get_sensor("wide")
v = env.agent
N = 50
t = time.time()
for _ in range(N):
  env.step([0.0, 0.5])
  road.perceive(False, v.origin, [0, 0, 1.2], [0, 0, 0])
  wide.perceive(False, v.origin, [0, 0, 1.2], [0, 0, 0])
  a, b = road.grab(), wide.grab()
print("step+2cams fps", N / (time.time() - t), a.shape, a.mean(), b.mean())
import cv2  # noqa: E402

cv2.imwrite(sys.argv[1], np.vstack([a, b])[..., ::-1])
env.close()
