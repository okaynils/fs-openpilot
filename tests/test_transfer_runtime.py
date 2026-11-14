"""Artifact-dependent integration gate; the Modal pilot supplies the real replay fixture."""
from pathlib import Path

import pytest

EPISODE = Path('data/sim/transfer_pilot_v1/val_1000_0')
EXPORT = Path('runs/transfer_pilot_analysis_v1/conditioned.onnx')
CHECKPOINT = Path('runs/transfer_pilot_conditioned_v1/ckpt/best.pt')


@pytest.mark.skipif(not EXPORT.exists() or not EPISODE.exists(), reason='requires completed Modal pilot artifacts')
def test_recorded_sequence_export_runtime_parity():
  from fsop.data.sim_dataset import Episode
  from fsop.eval.transfer import temporal_parity

  report = temporal_parity(CHECKPOINT, EXPORT, Episode(EPISODE), frames=12)
  assert report['plan_max_abs'] < .001
  assert report['hidden_max_abs'] < .00001


@pytest.mark.skipif(not CHECKPOINT.exists() or not EPISODE.exists(), reason='requires Modal pilot artifacts')
def test_context_ablation_reuses_only_unconditioned_vision():
  import numpy as np

  from fsop.data.sim_dataset import Episode
  from fsop.eval.open_loop import run_episode_model
  from fsop.geometry.camera import CameraMount
  from fsop.model.build import load_fs
  from fsop.model.conditioning import geometry_context
  from fsop.train.device import pick_device

  ep = Episode(EPISODE)
  ep.T = 3
  device = pick_device()
  model = load_fs(CHECKPOINT).to(device).eval()
  reference = run_episode_model(model, ep, device, bev=False, return_hidden=True)
  context = geometry_context(CameraMount(**ep.meta['calibrated_mount']), ep.vehicle)
  context[0] += .1
  fresh = run_episode_model(model, ep, device, bev=False, geometry_override=context)
  reused = run_episode_model(model, ep, device, bev=False, geometry_override=context,
                            encoded_hidden=reference['encoded_hidden'])
  np.testing.assert_array_equal(fresh['mu'], reused['mu'])
