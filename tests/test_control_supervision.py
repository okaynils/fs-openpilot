"""Training supervision must reproduce the actual deployed follower's coordinates and limits."""
import numpy as np
import torch

from fsop.eval.controller import PlanPurePursuit
from fsop.geometry.op_frames import ego_to_op_plan
from fsop.sim.expert import T_IDXS
from fsop.sim.vehicle import VehicleParams
from fsop.train.losses import follower_steering, steering_loss


def examples():
  vehicles = [VehicleParams(), VehicleParams(lf=.22, lr=.18, max_steer=.6)]
  plans, params, speeds, expected = [], [], [], []
  for vehicle in vehicles:
    for speed in (0., 1.3, 9.):
      for curvature in (-1.2, -.05, .05, 1.2):
        plan = np.zeros((33, 15), np.float32)
        plan[:, 0] = T_IDXS * speed
        plan[:, 1] = curvature * plan[:, 0]**2 / 2
        plans.append(ego_to_op_plan(plan))
        params.append([vehicle.wheelbase, vehicle.lr, vehicle.max_steer])
        speeds.append(speed)
        expected.append(PlanPurePursuit(vehicle)(plan, speed)[0])
  return torch.tensor(np.array(plans)), torch.tensor(speeds), torch.tensor(params), expected


def test_training_steering_matches_real_follower():
  plans, speeds, params, expected = examples()
  np.testing.assert_allclose(follower_steering(plans, speeds, params).numpy(), expected, atol=1e-6)


def test_control_loss_targets_near_term_action_and_has_finite_gradient():
  target = torch.zeros(2, 33, 15)
  target[:, :, 0] = torch.tensor(T_IDXS) * 1.5
  speeds = torch.tensor([1.5, 1.5])
  params = torch.tensor([[.4, .18, .6], [.4, .18, .6]])
  assert steering_loss(target, target, speeds, params) == 0
  far = target.clone()
  far[:, torch.tensor(T_IDXS) > 2., 1] = .1
  assert steering_loss(far, target, speeds, params) == 0
  near = target.clone()
  near[:, (torch.tensor(T_IDXS) > 0) & (torch.tensor(T_IDXS) < 1), 1] = .1
  near.requires_grad_()
  loss = steering_loss(near, target, speeds, params)
  assert loss > .01
  loss.backward()
  assert torch.isfinite(near.grad).all()
  assert near.grad[:, :, 1].abs().sum() > 0
