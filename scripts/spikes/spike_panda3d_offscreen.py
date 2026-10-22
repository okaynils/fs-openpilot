"""Spike: offscreen Panda3D rendering at 512x256 with an openpilot model-frame lens (macOS check)."""
import sys
import time

import numpy as np
from panda3d.core import (
  AmbientLight,
  DirectionalLight,
  FrameBufferProperties,
  GraphicsOutput,
  GraphicsPipe,
  PerspectiveLens,
  Texture,
  WindowProperties,
  loadPrcFileData,
)

loadPrcFileData("", "window-type offscreen\naudio-library-name null\nsync-video 0\n")
from direct.showbase.ShowBase import ShowBase  # noqa: E402

base = ShowBase()
print("pipe:", base.pipe.getInterfaceName(), base.win.getGsg().getDriverRenderer())
fb = FrameBufferProperties()
fb.setRgbColor(True)
fb.setRgbaBits(8, 8, 8, 0)
fb.setDepthBits(24)
buf = base.graphicsEngine.makeOutput(base.pipe, "cam", -2, fb, WindowProperties.size(512, 256),
                                     GraphicsPipe.BFRefuseWindow, base.win.getGsg(), base.win)
tex = Texture()
buf.addRenderTexture(tex, GraphicsOutput.RTMCopyRam)
cam = base.makeCamera(buf)
lens = PerspectiveLens()
lens.setFilmSize(512, 256)
lens.setFocalLength(910)
lens.setFilmOffset(0, 128 - 47.6)
lens.setNearFar(0.1, 500)
cam.node().setLens(lens)
cm = base.loader.loadModel("models/box") if False else None
from panda3d.core import CardMaker  # noqa: E402

c = CardMaker("ground")
c.setFrame(-50, 50, -50, 50)
g = base.render.attachNewNode(c.generate())
g.setP(-90)
g.setColor(0.35, 0.35, 0.37, 1)
for i in range(10):
  c2 = CardMaker("cone")
  c2.setFrame(-0.1, 0.1, 0, 0.33)
  n = base.render.attachNewNode(c2.generate())
  n.setPos(5 + 3 * i, 1.5 if i % 2 else -1.5, 0)
  n.setColor((0.1, 0.3, 0.9, 1) if i % 2 else (0.95, 0.8, 0.1, 1))
  n.setBillboardAxis()
dl = DirectionalLight("sun")
base.render.setLight(base.render.attachNewNode(dl))
al = AmbientLight("amb")
al.setColor((0.4, 0.4, 0.4, 1))
base.render.setLight(base.render.attachNewNode(al))
# Panda3D: x right, y forward, z up
cam.setPos(0, 0, 1.1)
cam.setHpr(0, 0, 0)
t = time.time()
N = 60
for i in range(N):
  cam.setPos(0, 0.1 * i, 1.1)
  base.graphicsEngine.renderFrame()
  img = np.frombuffer(tex.getRamImageAs("RGB").getData(), np.uint8).reshape(256, 512, 3)[::-1]
print("fps", N / (time.time() - t), img.shape, img.mean())
import cv2  # noqa: E402

cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "spike.png", img[..., ::-1])
