"""Export and replay the trained sequence-conditioned pilot through deployment queues."""
import hashlib
import json
from pathlib import Path

from fsop.data.sim_dataset import Episode
from fsop.eval.transfer import temporal_parity
from fsop.model.export import export_onnx

checkpoint = Path('runs/transfer_pilot_v2_retry1/ckpt/step_000300.pt')
root = Path('runs/transfer_pilot_verify_v2')
root.mkdir(exist_ok=True)
exported = export_onnx(checkpoint, root / 'conditioned.onnx')
rows = [dict(episode=name, **temporal_parity(checkpoint, exported,
        Episode(Path('data/sim/transfer_v2_pilot') / name), frames=8))
        for name in ['val_v2_0_expert', 'val_v2_2_expert']]
report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'export': str(exported), 'export_sha256': hashlib.sha256(exported.read_bytes()).hexdigest(), 'rows': rows}
(root / 'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
