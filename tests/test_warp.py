import numpy as np

from fsop.geometry.camera import CAMERA_MODELS, MEDMODEL_K
from fsop.geometry.warp import ModelFrameWarper, slot_maps
from fsop.profile import CameraSpec, TeamProfile


def _spec_like_model(**kw):
  K = MEDMODEL_K
  return CameraSpec(name="cam", width=512, height=256, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], **kw)


def test_identity_when_camera_is_the_model_camera():
  mx, my = slot_maps(_spec_like_model(), "road")
  u, v = np.meshgrid(np.arange(512), np.arange(256))
  np.testing.assert_allclose(mx, u, atol=1e-3)
  np.testing.assert_allclose(my, v, atol=1e-3)


def test_fisheye_point_lands_where_projection_says():
  import cv2

  from fsop.geometry.camera import VIEW_FROM_EGO
  from fsop.geometry.warp import _cam_rot

  cam = CameraSpec(name="cam", width=878, height=522, fx=426.0, fy=426.0, cx=439.0, cy=261.0, fisheye=True,
                   dist=[0.02, -0.01, 0.0, 0.0], pitch_deg=20.0, slots=["road", "wide"])
  p_body = np.array([12.0, -1.5, -1.0])
  pv = VIEW_FROM_EGO @ (_cam_rot(0.0, np.deg2rad(20.0), 0.0).T @ p_body)
  px = cv2.fisheye.distortPoints((pv[:2] / pv[2]).reshape(1, 1, 2), cam.K, np.array(cam.dist)).ravel()
  mx, my = slot_maps(cam, "wide")
  uv, _ = CAMERA_MODELS["wide"].project_cam(p_body[None])
  u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
  assert abs(mx[v, u] - px[0]) < 2.0 and abs(my[v, u] - px[1]) < 2.0


def test_point_lands_where_projection_says():
  """A real camera (different K, pitched 3 deg, with distortion) sees a 3D point; after warping, the point
  must appear where the model-slot camera would project it."""
  import cv2

  cam = CameraSpec(name="front", width=1920, height=1200, fx=1000.0, fy=1000.0, cx=950.0, cy=610.0,
                   dist=[-0.1, 0.02, 0.0, 0.0, 0.0], pitch_deg=3.0, slots=["road", "wide"])
  prof = TeamProfile(cameras=[cam])
  p_body = np.array([15.0, 1.2, -0.9])  # point in (virtual) camera body frame: x fwd, y left, z up
  # physical camera pixel
  from fsop.geometry.camera import VIEW_FROM_EGO
  from fsop.geometry.warp import _cam_rot

  R = _cam_rot(0.0, np.deg2rad(3.0), 0.0)
  pv = (VIEW_FROM_EGO @ (R.T @ p_body))
  px, _ = cv2.projectPoints(pv[None], np.zeros(3), np.zeros(3), cam.K, np.array(cam.dist))
  img = np.zeros((1200, 1920, 3), np.uint8)
  x, y = px.ravel()
  cv2.circle(img, (int(round(x)), int(round(y))), 4, (255, 255, 255), -1)
  out = ModelFrameWarper(prof)({"front": img})
  for slot in ("road", "wide"):
    uv, z = CAMERA_MODELS[slot].project_cam(p_body[None])
    ys, xs = np.nonzero(out[slot][..., 0] > 128)
    assert len(xs), slot
    assert abs(xs.mean() - uv[0, 0]) < 1.5 and abs(ys.mean() - uv[0, 1]) < 1.5, (slot, xs.mean(), ys.mean(), uv)


def test_project_to_camera_inverts_slot_maps():
  from fsop.geometry.camera import CameraMount
  from fsop.geometry.warp import project_to_camera

  cam = CameraSpec(name="cam", width=878, height=522, fx=426.0, fy=426.0, cx=439.0, cy=261.0, fisheye=True,
                   pitch_deg=15.0, slots=["wide"])
  T = CameraMount(z=1.2, pitch=np.deg2rad(21.0)).T_ego_cam()
  p = np.array([[9.0, 1.0, 0.0]])  # ground point, ego frame
  uv, front = project_to_camera(p, T, cam)
  Ti = np.linalg.inv(T)
  uv_slot, _ = CAMERA_MODELS["wide"].project_cam(p @ Ti[:3, :3].T + Ti[:3, 3])
  mx, my = slot_maps(cam, "wide")
  u, v = int(round(uv_slot[0, 0])), int(round(uv_slot[0, 1]))
  assert front[0] and abs(mx[v, u] - uv[0, 0]) < 2.0 and abs(my[v, u] - uv[0, 1]) < 2.0
