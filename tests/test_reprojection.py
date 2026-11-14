"""Rendered cones must appear where the pinhole model projects them (exact model-frame geometry)."""

import os

import numpy as np
import pytest

panda3d = pytest.importorskip("panda3d")
if os.environ.get("FSOP_SKIP_RENDER"):
  pytest.skip("rendering disabled (FSOP_SKIP_RENDER)", allow_module_level=True)

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


def test_physical_sensor_warp_translated_and_rotated():
  from fsop.geometry.camera import CameraModel
  from fsop.geometry.warp import ModelFrameWarper, _cam_rot
  from fsop.profile import CameraSpec, TeamProfile
  from fsop.sim.render.panda3d_backend import Panda3DRenderer

  spec = CameraSpec(width=640, height=480, fx=460, fy=420, cx=312, cy=238,
                    pitch_deg=3, roll_deg=-2, yaw_deg=2)
  model = CameraModel('front', spec.K, spec.width, spec.height)
  renderer = Panda3DRenderer({'front': model})
  xy = np.array([[12., 1.5], [20., -1.5]])
  renderer.load_scene(acceleration(), xy, [ConeColor.ORANGE] * 2,
                      SceneDR(grass_ring=False, sky_color=(0, 0, 0), ground_base=.1, ground_contrast=0))
  mount = CameraMount(x=-.4, y=.2, z=.55, pitch=.03, roll=.01, yaw=-.02)
  offset = np.eye(4)
  offset[:3, :3] = _cam_rot(*np.deg2rad([spec.roll_deg, spec.pitch_deg, spec.yaw_deg]))
  raw = renderer.render({'front': mount.T_ego_cam() @ offset})
  warped = ModelFrameWarper(TeamProfile(mount=mount, cameras=[spec]))(raw)
  for slot, image in warped.items():
    uv, _, visible = project_world(np.c_[xy, np.zeros(2)], np.eye(4), mount.T_ego_cam(), CAMERA_MODELS[slot])
    for (u, v), ok in zip(uv, visible, strict=True):
      if ok:
        patch = image[max(int(v)-35, 0):int(v)+4, max(int(u)-12, 0):int(u)+12]
        ys, xs = np.nonzero(patch.max(-1) > 60)
        assert len(ys) > 3
        assert abs(ys.max() + max(int(v)-35, 0) - v) <= 4
        assert abs((xs.min()+xs.max())/2 + max(int(u)-12, 0) - u) <= 4


def test_renderer_scenes_are_isolated_and_close_releases_resources():
  from fsop.sim.render.panda3d_backend import Panda3DRenderer

  dr = SceneDR(grass_ring=False, sky_color=(0, 0, 0), ground_base=.1, ground_contrast=0)
  pose = {name: CameraMount(z=1.1).T_ego_cam() for name in CAMERA_MODELS}
  first = Panda3DRenderer(CAMERA_MODELS)
  first.load_scene(acceleration(), np.array([[12., 1.5]]), [ConeColor.ORANGE], dr)
  before = first.render(pose)
  with Panda3DRenderer(CAMERA_MODELS) as second:
    second.load_scene(acceleration(), np.array([[5., -.5]]), [ConeColor.BLUE], dr)
    second.render(pose)
    after = first.render(pose)
    for name in before:
      np.testing.assert_array_equal(before[name], after[name])
    scene, buffers = second.scene, list(second.buffers.values())
  assert scene.isEmpty()
  assert all(not b.isActive() for b in buffers)
  assert not second.cam_nps and not second.buffers
  second.close()
  first.close()
