"""Create a diagnostic-only index for inspection of the reference-collection probe."""
import json
import runpy
import sys
from pathlib import Path

root = Path('data/sim/transfer_reference_collection_probe_v3')
report = json.loads((root/'report.json').read_text())
index = {'episodes': [report['result']]}
assert index['episodes'][0]['split'] == 'diagnostic'
path = root/'index.json'
if path.exists() and json.loads(path.read_text()) != index:
  raise ValueError('Preserve existing diagnostic index')
path.write_text(json.dumps(index,indent=2))
sys.argv = ['scripts/transfer_v2_preview.py',str(root)]
runpy.run_path(sys.argv[0],run_name='__main__')
