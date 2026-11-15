"""Reproduce the validation comparison from saved, paired experiment records."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root=Path('benchmarks/transfer_v2')):
  selection = json.loads((root / 'selection.json').read_text())
  ranking = {r['checkpoint']: r for r in selection['ranking']}
  labels, ade, passes, contacts = [], [], [], []
  for arm, label in [('expert', 'Expert data'), ('recovery', '+ assisted data'),
                     ('sequence', '+ history context'), ('policy', '+ policy training')]:
    history = [json.loads(line) for line in (root / f'transfer_{arm}_v2_validation.jsonl').read_text().splitlines()]
    for step in (500, 1500):
      labels.append(f'{label}\nstep {step:,}')
      ade.append(next(r['ade_4s'] for r in history if r['step'] == step))
      row = ranking[f'runs/transfer_{arm}_v2/ckpt/step_{step:06d}.pt']
      passes.append(row['passes'])
      contacts.append(row['contacts'])
  fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
  positions = np.arange(len(labels))
  axes[0].bar(positions, ade, color=['#337ea6'] * 6 + ['#a56630'] * 2)
  axes[0].set(ylabel='Recorded validation ADE (m)', ylim=(0, 2.7))
  axes[0].set_title('Lower recorded trajectory error did not improve driving transfer', loc='left')
  for x, value in zip(positions, ade, strict=True):
    axes[0].text(x, value + .05, f'{value:.2f}', ha='center')
  axes[1].bar(positions, passes, color=['#337ea6'] * 6 + ['#a56630'] * 2)
  axes[1].axhline(3, color='#333333', linestyle='--', label='Original baseline and previous candidate: 3/4')
  axes[1].set(ylabel='Driving validation passes / 4', ylim=(0, 4.2), yticks=range(5))
  axes[1].set_xticks(positions, labels, fontsize=9)
  axes[1].legend(loc='upper right', frameon=False)
  for x, value, count in zip(positions, passes, contacts, strict=True):
    axes[1].text(x, value + .08, f'{count} contacts', ha='center', fontsize=9)
  for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
  fig.savefig(root / 'validation_comparison.png', dpi=180)
  plt.close(fig)


if __name__ == '__main__':
  plot()
