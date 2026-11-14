"""Rejected resume attempts must not overwrite an experiment's original configuration."""
import json
from dataclasses import asdict

import pytest
import torch

from fsop.train.trainer import TrainCfg, Trainer


def test_invalid_resume_preserves_original_run(tmp_path):
  run = tmp_path / 'retained'
  run.mkdir()
  original = 'name: retained\nsteps: 2000\n'
  (run / 'config.yaml').write_text(original)
  checkpoint = run / 'derived.pt'
  torch.save({'inference_only': True}, checkpoint)
  cfg = TrainCfg(name='retained', out_dir=str(tmp_path), resume=str(checkpoint), device='cpu')
  with pytest.raises(ValueError, match='inference checkpoint'):
    Trainer(cfg)
  assert (run / 'config.yaml').read_text() == original
  assert sorted(p.name for p in run.iterdir()) == ['config.yaml', 'derived.pt']

  # A same-path dataset with changed geometry must also fail before model construction.
  dataset = tmp_path / 'dataset'
  dataset.mkdir()
  cfg.sim_root = str(dataset)
  (run / 'provenance.json').write_text(json.dumps({'dataset_manifest': {'geometry': 'original'}}))
  (dataset / 'manifest.json').write_text(json.dumps({'geometry': 'changed'}))
  torch.save({'cfg': asdict(cfg)}, checkpoint)
  with pytest.raises(ValueError, match='manifest differs'):
    Trainer(cfg)
  assert (run / 'config.yaml').read_text() == original
