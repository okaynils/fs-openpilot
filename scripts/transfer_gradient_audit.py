"""Fixed-sample gradient diagnostic; no optimizer updates or final-test access."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fsop.config import load_config
from fsop.data.sim_dataset import collate
from fsop.train.losses import steering_loss
from fsop.train.trainer import TrainCfg, Trainer


def run():
  cfg = load_config(TrainCfg, Path('configs/train/transfer_vision_last_pilot_v4.yaml'))
  cfg.name = 'transfer_gradient_audit_v6'
  cfg.num_workers = 0
  cfg.seed = 184195
  trainer = Trainer(cfg)
  model = trainer.model.eval()
  parameters = [p for p in model.base.vision.parameters() if p.requires_grad]
  indices = sorted(np.random.default_rng(cfg.seed).choice(len(trainer.train_ds), 32, replace=False).tolist())
  results = []
  for checkpoint in (cfg.init_ckpt, 'runs/transfer_vision_last_pilot_v4/ckpt/step_000100.pt'):
    # Trainer loads all initialization tensors; the trained checkpoint replaces them.
    if checkpoint != cfg.init_ckpt:
      model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    rows = []
    for start in range(0, len(indices), 8):
      selected = indices[start:start+8]
      batch = trainer._to(collate([trainer.train_ds[i] for i in selected]))
      out = trainer.forward(batch)
      losses = trainer.losses(out, batch)
      groups = {'plan': losses['plan'],
                'bev': sum(v for k, v in losses.items() if k.startswith('bev_')),
                'image': sum(v for k, v in losses.items() if k.startswith('img_')),
                'steering': steering_loss(out['plan_mu'], batch['plan'], batch['speed'], batch['control_vehicle'])}
      gradients = {}
      for name, loss in groups.items():
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        gradients[name] = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                                     for p, g in zip(parameters, grads, strict=True)]).detach()
      gradients['auxiliary'] = gradients['bev'] + gradients['image']
      gradients['total'] = gradients['plan'] + gradients['auxiliary']
      norms = {k: float(v.norm()) for k, v in gradients.items()}
      cosines = {}
      for left, right in [('plan', 'bev'), ('plan', 'image'), ('plan', 'auxiliary'),
                          ('steering', 'plan'), ('steering', 'total')]:
        denominator = norms[left] * norms[right]
        dot = float(torch.dot(gradients[left], gradients[right]))
        cosines[f'{left}:{right}'] = dot / denominator if denominator else None
      row = dict(indices=selected, losses={k: float(v.detach()) for k, v in groups.items()},
                 gradient_norms=norms, cosine=cosines,
                 wheelbases=batch['control_vehicle'][:, 0].detach().cpu().tolist())
      rows.append(row)
      print(json.dumps(row), flush=True)
      del out, losses, groups, gradients
    results.append(dict(checkpoint=checkpoint,
                        sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(), rows=rows))
  report = dict(seed=cfg.seed, indices=indices, precision='fp32', results=results,
                limitation='Local gradient alignment on 32 recorded training samples; not a causal closed-loop test. '
                           'Steering is a diagnostic only. L2-SP omitted from gradient groups; zero at initialization.')
  (trainer.run_dir/'report.json').write_text(json.dumps(report, indent=2))
  trainer.tb.close()


if __name__ == '__main__':
  run()
