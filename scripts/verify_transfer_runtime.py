"""Reproduce and isolate deployment parity on real recorded model inputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fsop.data.sim_dataset import Episode
from fsop.geometry.camera import CameraMount
from fsop.model.conditioning import geometry_context
from fsop.runtime import OnnxEngine, TorchEngine


def verify(checkpoint, exported, episode, output):
  ep = Episode(episode)
  geometry = geometry_context(CameraMount(**ep.meta.get('calibrated_mount', ep.meta['mount'])), ep.vehicle)
  args = (ep.frame_pair('road', 0), ep.frame_pair('wide', 0), np.zeros((24, 512), np.float32),
          np.zeros((25, 8), np.float32), np.array([1., 0.], np.float32), geometry)
  ort = OnnxEngine(exported, providers=['CPUExecutionProvider'])
  cuda = TorchEngine(checkpoint, device='cuda')
  # Deliberately recreate the historical reduced-precision setting for this ablation.
  # TorchEngine normally disables it; production inference must leave it disabled.
  torch.backends.cudnn.allow_tf32 = True
  result = {'providers': ort.sess.get_providers(), 'initial_cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'initial_matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'comparisons': {}}
  reference = ort(*args)

  def compare(engine):
    got = engine(*args)
    return {name: float(np.abs(got[name] - reference[name]).max()) for name in ('plan', 'hidden_state')}

  result['comparisons']['cuda_initial'] = compare(cuda)
  torch.backends.cudnn.allow_tf32 = False
  result['comparisons']['cuda_cudnn_tf32_off'] = compare(cuda)
  torch.backends.cuda.matmul.allow_tf32 = False
  result['comparisons']['cuda_all_tf32_off'] = compare(cuda)
  cpu = TorchEngine(checkpoint, device='cpu')
  result['comparisons']['cpu'] = compare(cpu)
  Path(output).write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2))
  assert result['comparisons']['cuda_all_tf32_off']['plan'] < .001
  assert result['comparisons']['cpu']['plan'] < .001
  return result


if __name__ == '__main__':
  verify('runs/transfer_pilot_conditioned_v1/ckpt/best.pt',
         'runs/transfer_pilot_analysis_v1/conditioned.onnx',
         'data/sim/transfer_pilot_v1/val_1000_0', 'runs/transfer_pilot_analysis_v1/parity_diagnosis_recheck.json')
