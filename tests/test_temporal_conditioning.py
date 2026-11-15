"""Geometry conditioning must be coherent across history and preserve padding/reference behavior."""
import numpy as np
import pytest
import torch

from fsop.model.build import SupercomboFS


def test_sequence_context_conditions_history_without_creating_fake_past():
  model = SupercomboFS(image_head=False, conditioned=True, anchored_context=True,
                       context_version=2, context_scope='sequence').eval()
  torch.nn.init.normal_(model.context_adapter[-1].weight, std=.01)
  torch.nn.init.normal_(model.context_adapter[-1].bias, std=.01)
  history = torch.randn(1, 24, 512)
  history[:, :4] = 0
  hidden = torch.randn(1, 512)
  captured = []

  def policy(features, desire, traffic):
    captured.append(features)
    return features.new_zeros((1, 990)), features.new_zeros((1, 8))

  model.base.policy_raw = policy
  anchor = model.context_anchor[None]
  context = anchor.clone()
  context[:, 2] -= .4
  model.plan(history, hidden, torch.zeros(1, 25, 8), torch.zeros(1, 2), context)
  actual = captured.pop()
  assert torch.count_nonzero(actual[:, :4]) == 0
  assert not torch.equal(actual[:, 4:24], history[:, 4:])
  assert not torch.equal(actual[:, 24], hidden)
  model.plan(history, hidden, torch.zeros(1, 25, 8), torch.zeros(1, 2), anchor)
  np.testing.assert_array_equal(captured.pop().detach().numpy(),
                                torch.cat([history, hidden[:, None]], 1).numpy())


def test_unknown_conditioning_scope_is_rejected():
  with pytest.raises(ValueError, match='scope'):
    SupercomboFS(conditioned=True, context_scope='unknown')
