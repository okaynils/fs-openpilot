"""Audit actual pilot observations, controls and expert projections before scaling."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from fsop.data.sim_dataset import Episode
from fsop.geometry.camera import CAMERA_MODELS, project_world
from fsop.geometry.yuv import model_to_rgb

root = Path(sys.argv[1])
panels = []
audit = []
for row in json.loads((root / 'index.json').read_text())['episodes']:
  if row.get('collector') not in ('expert', 'dagger'):
    continue
  ep = Episode(root / row['id'])
  for k in (min(10, ep.T-1), min(40, ep.T-1)):
    image = model_to_rgb(ep.frames['wide'][k]).copy()
    points = ep.plan[k, :, :3]
    uv, _, valid = project_world(points, np.eye(4), ep.camera_pose(k), CAMERA_MODELS['wide'])
    valid &= ep.observable(points, k, 'wide')
    for point in uv[valid]:
      cv2.circle(image, tuple(point.astype(int)), 3, (255, 60, 80), -1)
    cv2.putText(image, f"{row['id']} k={k} valid={ep.label_valid[k]}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    panels.append(cv2.resize(image, (512,256)))
  audit.append({'id': row['id'], 'max_pose_step': float(np.max(np.abs(np.diff(ep.camera_poses, axis=0)))),
                'max_origin_error': float(np.max(np.abs(ep.plan[:,0,:3]))),
                'recorded_contact_events': int(np.isfinite(ep.knock_times).sum()),
                'summary_contacts': ep.meta['summary']['doo'],
                'valid_frames': int(ep.label_valid.sum()), 'frames': ep.T})
mosaic = np.vstack([np.hstack(panels[i:i+4]) for i in range(0, len(panels), 4)])
cv2.imwrite(str(root / 'observations.jpg'), mosaic[..., ::-1])
(root / 'audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
