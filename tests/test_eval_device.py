"""An explicit evaluation backend must not silently select the host accelerator."""
import torch

from fsop.eval.closed_loop import ModelPolicy


def test_checkpoint_device_override_reaches_actual_policy_constructor(monkeypatch):
  monkeypatch.setattr('fsop.model.build.load_fs', lambda path: torch.nn.Linear(2, 2))
  def unexpected_auto():
    raise AssertionError('Explicit CPU evaluation must bypass automatic accelerator selection')
  monkeypatch.setattr('fsop.eval.closed_loop.pick_device', unexpected_auto)
  policy = ModelPolicy.from_checkpoint('unused.pt', device=torch.device('cpu'))
  assert policy.device == torch.device('cpu')
  assert next(policy.model.parameters()).device.type == 'cpu'
