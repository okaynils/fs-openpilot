from dataclasses import replace

import numpy as np

from fsop.geometry.camera import CAMERA_MODELS, CameraMount, project_world
from fsop.geometry.mounting import MountDynamics, body_image_mask, body_occluded
from fsop.model.conditioning import geometry_context
from fsop.profile import TeamProfile, load_profile
from fsop.sim.vehicle import VehicleParams


def test_translated_rotated_ground_projection():
  from fsop.eval.img_to_bev import ground_points

  points = np.array([[5., 1., 0.], [12., -2., 0.], [25., .5, 0.]])
  for height in (.25, .55, 1.65):
    for x, y in ((-1., -.3), (1.2, .4)):
      pose = CameraMount(x=x, y=y, z=height, roll=.03, pitch=.04, yaw=-.05).T_ego_cam()
      for name, cam in CAMERA_MODELS.items():
        uv, _, _ = project_world(points, np.eye(4), pose, cam)
        xy, valid = ground_points(uv, name, pose)
        assert valid.all()
        np.testing.assert_allclose(xy, points[:, :2], atol=1e-8)


def test_motion_is_continuous_and_observation_cadence_independent():
  motion = MountDynamics(.003, .3, 1.2, [1, 2, 3, 4, 5, 6])
  mount = CameraMount(x=-.2, y=.1, z=.3)
  poses = np.stack([motion.pose(mount, t).T_ego_cam() for t in np.arange(0, 2, .05)])
  assert np.abs(np.diff(poses[:, :3, 3], axis=0)).max() < .002
  np.testing.assert_array_equal(motion.pose(mount, .2).T_ego_cam(), poses[4])
  np.testing.assert_array_equal(mount.T_ego_cam(), MountDynamics().pose(mount, 100).T_ego_cam())


def test_body_mask_matches_point_visibility():
  vp = VehicleParams(body_height=.3)
  pose = CameraMount(x=-1, z=.6).T_ego_cam()
  camera = CAMERA_MODELS['wide']
  mask = body_image_mask(camera, pose, vp)
  points = np.array([[2., 0., 0.], [10., 3., 0.], [20., -3., 0.]])
  uv, _, visible = project_world(points, np.eye(4), pose, camera)
  blocked = body_occluded(points, pose, vp)
  for pixel, valid, occluded in zip(uv, visible, blocked, strict=True):
    if valid:
      u, v = np.rint(pixel).astype(int)
      assert mask[v, u] == occluded
  assert mask.any()
  assert not body_occluded(points, pose, replace(vp, body_height=0)).any()


def test_profile_motion_and_calibration_roundtrip(tmp_path):
  import yaml

  profile = TeamProfile(mount=CameraMount(x=-.3, z=.4), motion=MountDynamics(.002, .2),
                        calibration_error=CameraMount(z=.01, pitch=.005))
  path = tmp_path / 'profile.yaml'
  path.write_text(yaml.safe_dump(profile.to_dict()))
  loaded = load_profile(path)
  assert loaded.to_dict() == profile.to_dict()
  np.testing.assert_array_equal(geometry_context(loaded.mount, loaded.vehicle),
                                geometry_context(profile.mount, profile.vehicle))


def test_distribution_is_deterministic_and_coherent():
  from fsop.sim.distribution import sample_profile

  for seed in range(40):
    p, identity = sample_profile('configs/distributions/transfer_v1.yaml', seed)
    q, same = sample_profile('configs/distributions/transfer_v1.yaml', seed)
    assert p.to_dict() == q.to_dict() and identity == same
    assert p.mount.z > p.vehicle.body_height
    assert abs(p.vehicle.length - (p.vehicle.wheelbase + p.vehicle.front_overhang
                                  + p.vehicle.rear_overhang)) < 1e-10


def test_camera_translation_does_not_move_trajectory_origin():
  from fsop.sim.env import FSEnv, SimCfg
  from fsop.track.fixed_layouts import acceleration

  config = SimCfg(render=False, dr='off', randomize_expert=False)
  a = FSEnv(acceleration(), config, profile=TeamProfile())
  b = FSEnv(acceleration(), config, profile=TeamProfile(mount=CameraMount(x=-1, y=.3, z=.25)))
  np.testing.assert_array_equal(a.label(), b.label())
  np.testing.assert_allclose(a.label()[0, :3], 0, atol=1e-8)


def test_rover_expert_obeys_profile_limits():
  from fsop.sim.env import FSEnv, SimCfg
  from fsop.track.fixed_layouts import acceleration

  profile = load_profile('configs/profiles/generic_rover.yaml')
  env = FSEnv(acceleration(), SimCfg(render=False, dr='off'), profile=profile)
  assert env.expert_params.v_cap <= profile.vehicle.max_speed
  assert env.expert_params.a_brk <= profile.vehicle.max_brake
  assert env.expert_params.a_acc <= profile.vehicle.max_accel
  assert np.max(env.label()[:, 3]) <= profile.vehicle.max_speed + .05


def test_conditioned_identity_and_legacy_contract():
  import pytest
  import torch

  from fsop.model.build import SupercomboFS

  torch.manual_seed(19)
  legacy = SupercomboFS(image_head=False).eval()
  candidate = SupercomboFS(image_head=False, conditioned=True).eval()
  missing, unexpected = candidate.load_state_dict(legacy.state_dict(), strict=False)
  assert missing and all(k.startswith('context_adapter.') for k in missing) and not unexpected
  history, hidden = torch.randn(2, 24, 512), torch.randn(2, 512)
  desire, traffic = torch.zeros(2, 25, 8), torch.zeros(2, 2)
  context = torch.from_numpy(np.stack([geometry_context(CameraMount(), VehicleParams())] * 2))
  with torch.no_grad():
    ref = legacy.plan(history, hidden, desire, traffic)
    got = candidate.plan(history, hidden, desire, traffic, context)
  torch.testing.assert_close(ref['plan_mu'], got['plan_mu'], rtol=0, atol=0)
  with pytest.raises(ValueError, match='requires calibrated geometry'):
    candidate.plan(history, hidden, desire, traffic)


def test_images_cannot_identify_vehicle_origin_without_translation():
  from fsop.geometry.camera import se2_to_se3

  points = np.array([[8., 2., 0.], [14., -2., 0.]])
  ego_a, camera_a = np.eye(4), CameraMount(z=.4).T_ego_cam()
  ego_b = se2_to_se3(1., .2, 0.)
  camera_b = CameraMount(x=-1., y=-.2, z=.4).T_ego_cam()
  for camera in CAMERA_MODELS.values():
    uv_a = project_world(points, ego_a, camera_a, camera)[0]
    uv_b = project_world(points, ego_b, camera_b, camera)[0]
    np.testing.assert_array_equal(uv_a, uv_b)
  # Identical camera images correspond to different metric positions in the vehicle frame.
  assert not np.array_equal(points, points - ego_b[:3, 3])


def test_rover_controller_and_mission_assistance_respect_limits():
  from fsop.eval.closed_loop import mission_brake
  from fsop.eval.controller import PlanPurePursuit
  from fsop.sim.env import FSEnv, SimCfg
  from fsop.track.fixed_layouts import acceleration

  profile = load_profile('configs/profiles/generic_rover.yaml')
  profile.vehicle.max_speed = 1.
  env = FSEnv(acceleration(), SimCfg(render=False, dr='off'), profile=profile)
  env.state.vx = .8
  assert mission_brake(-.2, env.observe(), env) == -.2  # launch must not force a 1 m/s rover to 2 m/s
  plan = np.zeros((33, 15))
  plan[:, 0], plan[:, 1], plan[:, 3] = .5, 1., 20.
  steer, accel = PlanPurePursuit(profile.vehicle)(plan, 1.5)
  assert abs(steer) <= profile.vehicle.max_steer and accel < 0
  env.vp.max_brake = 1.
  env.s = env.mission.s_end
  assert mission_brake(0., env.observe(), env) >= -1.


def test_renderer_lens_matches_noncentral_anisotropic_intrinsics():
  import pytest

  core = pytest.importorskip('panda3d.core')
  from fsop.geometry.camera import CameraModel
  from fsop.sim.render.panda3d_backend import camera_lens

  camera = CameraModel('sensor', np.array([[460., 0, 312], [0, 420, 238], [0, 0, 1]]), 640, 480)
  lens = camera_lens(camera)
  for point in np.array([[12., 1.3, -.55], [20., -1.7, .8], [2., 0., 0.]]):
    screen = core.Point2()
    lens.project(core.Point3(-point[1], point[0], point[2]), screen)
    actual = np.array([(screen.x + 1) * camera.width / 2, (1 - screen.y) * camera.height / 2])
    expected = camera.project_cam(point[None])[0][0]
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_knocked_cones_are_removed_at_the_recorded_time(tmp_path):
  import json

  from fsop.data.sim_dataset import Episode

  (tmp_path / 'meta.json').write_text(json.dumps({'frames': 2, 'mount': {}, 'vehicle': {}}))
  states = np.zeros((2, 12))
  states[1, 0] = 1.
  for name, data in [('states', states), ('plan', np.zeros((2, 33, 15))), ('desire', np.zeros((2, 8))),
                      ('cones', np.array([[10., 0., 0.], [15., 1., 1.]]))]:
    np.save(tmp_path / f'{name}.npy', data)
  (tmp_path / 'events.jsonl').write_text(json.dumps({'kind': 'doo', 'cone': 0, 't': .5}) + '\n')
  episode = Episode(tmp_path)
  assert len(episode.cones_ego(0, visible_only=False)[0]) == 2
  assert len(episode.cones_ego(1, visible_only=False)[0]) == 1


def test_anchored_adapter_preserves_policy_at_reference_after_learning():
  import torch

  from fsop.model.build import SupercomboFS

  model = SupercomboFS(image_head=False, conditioned=True, anchored_context=True).eval()
  model.base.policy.requires_grad_(False)
  torch.nn.init.normal_(model.context_adapter[-1].weight, std=.03)
  torch.nn.init.normal_(model.context_adapter[-1].bias, std=.03)
  hidden, history = torch.randn(1, 512), torch.randn(1, 24, 512)
  desire, traffic = torch.zeros(1, 25, 8), torch.tensor([[1., 0.]])
  geometry = model.context_anchor[None].clone()
  with torch.no_grad():
    reference, _ = model.base.policy_raw(torch.cat([history, hidden[:, None]], 1), desire, traffic)
    actual = model.plan(history, hidden, desire, traffic, geometry)
  np.testing.assert_array_equal(actual['plan_mu'].numpy().ravel(), reference[0, :495].numpy())
  geometry[:, 2] -= .5
  result = model.plan(history, hidden, desire, traffic, geometry)
  assert not torch.equal(result['plan_mu'], actual['plan_mu'])
  result['plan_mu'].sum().backward()
  assert model.context_adapter[-1].weight.grad.abs().sum() > 0
  assert all(p.grad is None for p in model.base.policy.parameters())
  from fsop.model.conditioning import CONTEXT_V2_INDICES, prune_orientation_state
  pruned = SupercomboFS(image_head=False, conditioned=True, anchored_context=True, context_version=2).eval()
  pruned.load_state_dict(prune_orientation_state(model.state_dict()), strict=True)
  with torch.no_grad():
    specialized = pruned.plan(history, hidden, desire, traffic, geometry[:, list(CONTEXT_V2_INDICES)])
  torch.testing.assert_close(specialized['plan_mu'], result['plan_mu'], atol=.001, rtol=1e-5)
  assert pruned.context_adapter[0].in_features == 512 + 13


def test_distribution_identity_includes_referenced_profile(tmp_path):
  from pathlib import Path

  import yaml

  from fsop.sim.distribution import sample_profile

  profile = yaml.safe_load(Path('configs/profiles/generic_rover.yaml').read_text())
  profile_path = tmp_path / 'rover.yaml'
  profile_path.write_text(yaml.safe_dump(profile))
  distribution = yaml.safe_load(Path('configs/distributions/transfer_v1.yaml').read_text())
  distribution['families']['rover']['profile'] = str(profile_path)
  path = tmp_path / 'distribution.yaml'
  path.write_text(yaml.safe_dump(distribution))
  before, old = sample_profile(path, 12, 'rover')
  profile['vehicle']['lf'] *= 1.05
  profile_path.write_text(yaml.safe_dump(profile))
  after, new = sample_profile(path, 12, 'rover')
  assert old['sha256'] == new['sha256']
  assert old['profile_sha256'] != new['profile_sha256']
  assert before.vehicle.wheelbase != after.vehicle.wheelbase
