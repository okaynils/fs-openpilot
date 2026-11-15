"""Measured speed must reach prediction without changing legacy checkpoint behavior."""
import pytest
import torch

from fsop.model.build import SupercomboFS, load_fs
from fsop.model.conditioning import context_fields


@pytest.mark.parametrize('mode', ['speed', 'constant'])
def test_speed_adapter_zero_initialization_and_input_information(mode, tmp_path):
  torch.manual_seed(194)
  model = SupercomboFS(image_head=False, conditioned=True, context_version=2, state_mode=mode).eval()
  history, hidden = torch.randn(1, 24, 512), torch.randn(1, 512)
  desire, traffic, geometry = torch.zeros(1, 25, 8), torch.ones(1, 2), torch.randn(1, 13)
  captured = []

  def policy(features, desire, traffic):
    captured.append(features.detach().clone())
    return features.new_zeros((1, 990)), features.new_zeros((1, 8))

  model.base.policy_raw = policy
  model.plan(history, hidden, desire, traffic, geometry, torch.tensor([1.]))
  torch.testing.assert_close(captured.pop(), torch.cat([history, hidden[:, None]], 1), rtol=0, atol=0)
  torch.nn.init.normal_(model.motion_adapter[-1].weight, std=.1)
  for speed in (0., 2.):
    model.plan(history, hidden, desire, traffic, geometry, torch.tensor([speed]))
  first, second = captured
  torch.testing.assert_close(first[:, :-1], second[:, :-1], rtol=0, atol=0)
  assert torch.equal(first[:, -1], second[:, -1]) == (mode == 'constant')
  if mode == 'speed':
    with pytest.raises(ValueError, match='speed'):
      model.plan(history, hidden, desire, traffic, geometry)
  # Weight-only artifacts cannot silently acquire a new input contract.
  path = tmp_path/'bad.pt'
  torch.save({'model': model.state_dict(), 'input_contract': {
    'geometry_version': 2, 'fields': list(context_fields(2)),
    'anchored_context': False}}, path)
  with pytest.raises(ValueError, match='motion|state'):
    load_fs(path)


def test_invalid_state_mode_is_rejected_before_loading_graphs():
  with pytest.raises(ValueError, match='state'):
    SupercomboFS(state_mode='unknown')
