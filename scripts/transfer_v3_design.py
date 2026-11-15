"""Freeze new unseen final configurations before the controller interaction probe."""
from pathlib import Path

import yaml
from transfer_v2_experiment import configuration


def write_design():
  choices = [('formula', (.76, .18, .32), 'interpolation'),
             ('formula', (.16, .73, .91), 'interpolation'),
             ('rover', (.83, .77, .09), 'interpolation'),
             ('rover', (.14, .22, .86), 'interpolation'),
             ('formula', (-.08, 1.08, 1.12), 'extrapolation'),
             ('rover', (1.08, -.08, -.03), 'extrapolation')]
  configurations = [configuration(484191+i, family, fractions, f'test_v3_{i}', regime)
                    for i, (family, fractions, regime) in enumerate(choices)]
  suite = {'version': 2, 'experiment': 'transfer_v3', 'track_seeds': [99211, 99212],
           'scene_seeds': [6201, 6202], 'configurations': configurations}
  path = Path('configs/eval/transfer_v3_test.yaml')
  text = yaml.safe_dump(suite, sort_keys=False)
  if path.exists() and path.read_text() != text:
    raise ValueError('Preserve frozen design')
  path.write_text(text)


if __name__ == '__main__':
  write_design()
