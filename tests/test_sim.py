import numpy as np

from fsop.geometry.op_frames import ego_to_op_plan
from fsop.sim.env import FSEnv, SimCfg
from fsop.sim.expert import plan_from_rollout
from fsop.sim.referee import _seg_intersect
from fsop.track.fixed_layouts import acceleration, skidpad
from fsop.track.generate import generate_track


def _run(track, noise=False, seed=0):
  cfg = SimCfg(render=False)
  cfg.noise.enabled = noise
  env = FSEnv(track, cfg, seed=seed)
  while not env.done:
    env.step(*env.expert_action())
  return env


def test_expert_passes_all_disciplines():
  for t in (generate_track(2), skidpad(), acceleration()):
    t.laps = min(t.laps, 2) if t.discipline == "trackdrive" else t.laps
    s = _run(t).referee.summary()
    assert s["pass"], s
    assert s["doo"] == 0 and s["oc"] == 0


def test_expert_offset_tracks_parallel_line():
  env = FSEnv(acceleration(width=5.0), SimCfg(render=False), seed=0)
  for _ in range(120):
    env.offset_val = 0.8  # recovery excursion target, left of the centreline
    env.step(*env.expert_action(), add_noise=False)
  assert abs(env.lat - 0.8) < 0.1, env.lat
  plan = env.label()  # labels ignore the excursion: they steer back towards the centreline
  assert plan[16, 1] < -0.4


def test_plan_label_straight_and_left_turn():
  t = np.arange(0, 10.001, 0.02)
  v = 10.0
  straight = np.c_[v * t, 0 * t, 0 * t, np.full_like(t, v), 0 * t, 0 * t, 0 * t, 0 * t, t]
  p = plan_from_rollout(straight)
  np.testing.assert_allclose(p[:, 0], v * np.array([10 * (i / 32) ** 2 for i in range(33)]), atol=1e-6)
  assert np.abs(p[:, 1]).max() < 1e-9
  r = 0.5  # left turn, yaw rate 0.5 rad/s
  yaw = r * t
  x = v / r * np.sin(yaw)
  y = v / r * (1 - np.cos(yaw))
  turn = np.c_[x, y, yaw, np.full_like(t, v), 0 * t, np.full_like(t, r), 0 * t, 0 * t, t]
  p = plan_from_rollout(turn)
  assert p[10, 1] > 0 and p[10, 11] > 0 and p[0, 14] > 0  # left = +y in ego frame
  q = ego_to_op_plan(p)
  assert q[10, 1] < 0 and q[10, 11] < 0  # openpilot frame: y right, yaw clockwise


def test_segment_intersection():
  assert _seg_intersect((0, -1), (0, 1), (-1, 0), (1, 0))
  assert not _seg_intersect((0, 1), (0, 2), (-1, 0), (1, 0))


def test_cone_hit_detected():
  t = acceleration()
  cfg = SimCfg(render=False)
  cfg.noise.enabled = False
  env = FSEnv(t, cfg, seed=0)
  for _ in range(200):  # steer hard left into the blue cones
    env.step(0.3, 3.0, add_noise=False)
  assert env.referee.n_doo > 0


def test_image_keypoint_ground_projection_roundtrip():
  from fsop.eval.img_to_bev import ground_points
  from fsop.geometry.camera import CAMERA_MODELS, CameraMount, project_world

  T = CameraMount(z=1.05, pitch=0.03, roll=-0.01, yaw=0.01).T_ego_cam()
  pts = np.array([[8.0, 2.0], [20.0, -1.5], [35.0, 0.3]])
  for cam in ("road", "wide"):
    uv, _, _ = project_world(np.c_[pts, np.zeros(3)], np.eye(4), T, CAMERA_MODELS[cam])
    xy, ok = ground_points(uv, cam, T)
    assert ok.all()
    np.testing.assert_allclose(xy, pts, atol=1e-6)


def test_frames_sharded_roundtrip_and_repack(tmp_path):
  import zarr

  from fsop.sim.recorder import repack_frames, write_frames

  rng = np.random.default_rng(0)
  a = rng.integers(0, 255, (7, 6, 128, 256), dtype=np.uint8)
  write_frames(tmp_path / "ep" / "frames.zarr", {"road": a})
  g = zarr.open_group(str(tmp_path / "ep" / "frames.zarr"), mode="r")
  assert g["road"].shards is not None and np.array_equal(g["road"][3], a[3])
  # legacy layout (one file per frame) -> sharded, bit-exact, idempotent
  old = zarr.group(store=zarr.storage.LocalStore(str(tmp_path / "old" / "frames.zarr")), overwrite=True)
  old.create_array("wide", shape=a.shape, chunks=(1, 6, 128, 256), dtype="uint8")[:] = a
  assert repack_frames(tmp_path / "old") and not repack_frames(tmp_path / "old")
  g = zarr.open_group(str(tmp_path / "old" / "frames.zarr"), mode="r")
  assert g["wide"].shards is not None and np.array_equal(g["wide"][:], a)
