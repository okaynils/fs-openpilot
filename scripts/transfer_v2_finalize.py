"""Freeze the driving selection before opening the independent test suite."""
import json
from pathlib import Path

from transfer_v2_merge import merge
from transfer_v2_package import package
from transfer_v2_select import rank


def finalize():
  root = Path('runs/transfer_validation_combined_v2')
  merge('runs/transfer_validation_v2/report.json',
        'runs/transfer_policy_validation_v2/report.json', root / 'report.json')
  selection = rank(json.loads((root / 'report.json').read_text()))
  (root / 'selection.json').write_text(json.dumps(selection, indent=2))
  print(json.dumps(selection, indent=2), flush=True)
  package(root / 'selection.json', 'runs/transfer_selected_v2')


if __name__ == '__main__':
  finalize()
