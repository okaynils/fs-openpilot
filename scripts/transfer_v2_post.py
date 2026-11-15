"""Run the fixed post-selection default, same-state and calibration checks."""
from transfer_v2_metrics import metrics

if __name__ == '__main__':
  for stage in ('runtime_suite', 'default', 'test', 'context'):
    print(f'Post-selection stage: {stage}', flush=True)
    metrics(stage)
