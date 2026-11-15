"""Inspect every paired collection episode without training or rerendering."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from fsop.data.sim_dataset import Episode
from fsop.geometry.camera import CAMERA_MODELS, project_world
from fsop.geometry.yuv import model_to_rgb

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/sim/transfer_refresh_pilot_v7')


def run():
  report = json.loads((ROOT/'report.json').read_text())
  audit = []
  for row in report['rows']:
    ep = Episode(ROOT/row['id'])
    takeover = np.load(ep.path/'expert_intervention.npy')
    controls = np.load(ep.path/'controls.npy')
    keys = sorted(set([0, min(10, ep.T-1), min(40, ep.T-1), ep.T//2, ep.T-1]
                      + np.flatnonzero(~ep.label_valid)[:1].tolist()))
    panels = []
    for k in keys:
      image = model_to_rgb(ep.frames['wide'][k]).copy()
      points = ep.plan[k, :, :3]
      uv, _, valid = project_world(points, np.eye(4), ep.camera_pose(k), CAMERA_MODELS['wide'])
      valid &= ep.observable(points, k)
      for pixel in uv[valid]:
        cv2.circle(image, tuple(pixel.astype(int)), 3, (255, 60, 80), -1)
      label = f"{row['id']} t={ep.states[k, 0]:.1f} expert={int(takeover[k])} valid={int(ep.label_valid[k])}"
      cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
      panels.append(image)
    cv2.imwrite(str(ROOT/f"{row['id']}.jpg"), np.vstack(panels)[..., ::-1])
    audit.append(dict(id=row['id'], frames=keys,
                      first_invalid_time=float(ep.states[np.flatnonzero(~ep.label_valid)[0], 0])
                      if (~ep.label_valid).any() else None,
                      samples=[dict(time=float(ep.states[k, 0]), speed=float(ep.states[k, 4]),
                                    expert_control=bool(takeover[k]), valid=bool(ep.label_valid[k]),
                                    controls=controls[k].tolist()) for k in range(ep.T)],
                      events=[json.loads(line) for line in (ep.path/'events.jsonl').read_text().splitlines()]))
  quality = dict(valid_each_arm=all(r['valid_fraction'] >= .95 for r in report['rows']),
                 contacts_per_episode=all(r['contacts'] <= 2 for r in report['rows']))
  rover = [r for r in report['rows'] if r['id'] in ('train_v2_03_current', 'train_v2_13_current')]
  quality['rover_expert_fraction'] = sum(r['expert_control_fraction'] for r in rover)/len(rover) < .75
  (ROOT/'inspection.json').write_text(json.dumps(dict(quality=quality, episodes=audit), indent=2))
  (ROOT/'inspection_source.py').write_bytes(Path(__file__).read_bytes())
  print(json.dumps(quality), flush=True)


if __name__ == '__main__':
  run()
