"""Rendered cones must appear where the pinhole model projects them (exact model-frame geometry)."""

import numpy as np
import pytest

panda3d = pytest.importorskip("panda3d")

from fsop.geometry.camera import CAMERA_MODELS, CameraMount, project_world, se2_to_se3  # noqa: E402
from fsop.sim.randomization import SceneDR  # noqa: E402
from fsop.track.fixed_layouts import acceleration  # noqa: E402
from fsop.track.schema import ConeColor  # noqa: E402


def test_cone_base_reprojects():
  from fsop.sim.render.panda3d_backend import Panda3DRenderer

  t = acceleration()
  xy = np.array([[12.0, 1.5], [20.0, -1.5]])
  r = Panda3DRenderer(CAMERA_MODELS)
  dr = SceneDR(grass_ring=False, sky_color=(0.0, 0.0, 0.0), ground_base=0.1, ground_contrast=0.0)
  r.load_scene(t, xy, [ConeColor.ORANGE, ConeColor.ORANGE], dr)
  T_we = se2_to_se3(0.0, 0.0, 0.0)
  T_ec = CameraMount(z=1.1).T_ego_cam()
  imgs = r.render({"road": T_we @ T_ec, "wide": T_we @ T_ec})
  for name, img in imgs.items():
    uv, z, vis = project_world(np.c_[xy, np.zeros(2)], T_we, T_ec, CAMERA_MODELS[name])
    orange = img.max(-1) > 60  # dark ground + black sky: anything bright is cone
    for (u, v), ok in zip(uv, vis, strict=True):
      if not ok:
        continue
      win = orange[max(int(v) - 40, 0):int(v) + 3, max(int(u) - 15, 0):int(u) + 15]
      ys, xs = np.nonzero(win)
      assert len(ys) > 3, f"cone not rendered near projection in {name}"
      bottom = ys.max() + max(int(v) - 40, 0)
      center = (xs.min() + xs.max()) / 2 + max(int(u) - 15, 0)
      assert abs(bottom - v) <= 3.0 and abs(center - u) <= 3.0, (name, bottom, v, center, u)
