"""Preserve a driving-selected candidate and verify its exported temporal runtime."""
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import tarfile
from pathlib import Path

import torch

from fsop.data.sim_dataset import Episode
from fsop.eval.transfer import temporal_parity
from fsop.model.export import export_onnx


def package(selection_path, output):
  selection_path, root = Path(selection_path), Path(output)
  selection = json.loads(selection_path.read_text())
  source = Path(selection['selected_candidate'])
  if root.exists():
    raise ValueError('Preserve existing package; use a distinct output directory')
  root.mkdir(parents=True)
  checkpoint = root / 'selected.pt'
  shutil.copy2(source, checkpoint)
  shutil.copy2(selection_path, root / 'selection.json')
  exported = export_onnx(checkpoint, root / 'candidate.onnx')
  parity = [dict(episode=eid, **temporal_parity(checkpoint, exported,
            Episode(Path('data/sim/transfer_train_v2') / eid), frames=30))
            for eid in ('val_v2_0_expert', 'val_v2_2_expert')]
  with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
    for path in ('src', 'configs', 'scripts', 'tests', 'pyproject.toml', 'uv.lock'):
      archive.add(path, filter=lambda item: None if '__pycache__' in item.name else item)
  saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
  report = {'source_checkpoint': str(source), 'selected_step': saved['step'],
            'input_contract': saved['input_contract'], 'parity': parity,
            'packaging_environment': {'python': platform.python_version(),
                                      'packages': {name: importlib.metadata.version(name)
                                                   for name in ('torch', 'numpy', 'onnx', 'onnxruntime',
                                                                'opencv-python-headless', 'panda3d')}},
            'artifacts': {p.name: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                   'bytes': p.stat().st_size}
                          for p in (checkpoint, exported, root/'selection.json', root/'source.tar.gz')},
            'status': 'research candidate; promotion requires default and independent final driving results',
            'runtime_requirement': 'use accompanying source; v1 PyTorch loaders lack sequence conditioning'}
  (root / 'artifacts.json').write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2))
  return report


if __name__ == '__main__':
  package(sys.argv[1], sys.argv[2])
