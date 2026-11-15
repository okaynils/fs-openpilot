"""Post-selection metrics; never use independent test results for selection."""
import json
import subprocess
import sys
from pathlib import Path

from fsop.eval.transfer import context_ablation, paired_open_loop

CHECKPOINT = 'runs/transfer_selected_v2/selected.pt'


def metrics(stage):
  root = Path('runs/transfer_selected_v2')
  if not (root / 'selection.json').exists():
    raise ValueError('Freeze selection before evaluation')
  if stage == 'default':
    subprocess.run([sys.executable, '-m', 'fsop.cli', 'eval', 'open-loop',
                    '--dataset', 'data/sim/v1', '--ckpt', CHECKPOINT,
                    '--out', str(root / 'default_open.json')], check=True)
    subprocess.run([sys.executable, '-m', 'fsop.cli', 'eval', 'closed-loop',
                    '--ckpt', CHECKPOINT, '--suite', 'configs/eval/closed_loop_quick.yaml',
                    '--out', str(root / 'default_closed'), '--no-rrd'], check=True)
  elif stage == 'context':
    context_ablation('data/sim/transfer_train_v2', CHECKPOINT, root / 'context_ablation.json')
  elif stage == 'test':
    paired_open_loop('data/sim/transfer_test_v2',
                     ['runs/cuda_racing_v6/ckpt/best.pt', 'runs/transfer_minimal_v1/ckpt/best.pt', CHECKPOINT],
                     root / 'final_open.json')
  elif stage == 'runtime_suite':
    import yaml
    suite = yaml.safe_load(Path('configs/eval/transfer_v2_test.yaml').read_text())
    # Fixed first Formula and first rover, first seed: declared before test observations.
    suite['configurations'] = [suite['configurations'][i] for i in (0, 2)]
    suite['track_seeds'], suite['scene_seeds'] = suite['track_seeds'][:1], suite['scene_seeds'][:1]
    # The generic single-seed suite alternates recovery by configuration index.
    (root / 'runtime_suite.yaml').write_text(yaml.safe_dump(suite, sort_keys=False))
    print(json.dumps({'runtime_configurations': [c['name'] for c in suite['configurations']],
                      'recovery': [False, True]}))
  else:
    raise ValueError(stage)


if __name__ == '__main__':
  metrics(sys.argv[1])
