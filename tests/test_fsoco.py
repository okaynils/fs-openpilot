"""FSOCO importer + dataset on a synthetic Supervisely-format export (real FSOCO is not redistributable)."""

import json

import cv2
import numpy as np

from fsop.data.fsoco import FsocoDataset, import_fsoco


def _make_sly(root, teams=("teamA", "teamB", "teamC", "teamD"), n=3, border=20):
  root.mkdir(parents=True, exist_ok=True)
  (root / "meta.json").write_text(json.dumps({"classes": []}))
  rng = np.random.default_rng(0)
  for t in teams:
    (root / t / "img").mkdir(parents=True)
    (root / t / "ann").mkdir(parents=True)
    for i in range(n):
      H, W = 720 + 2 * border, 1280 + 2 * border
      img = np.full((H, W, 3), 90, np.uint8)
      objs = []
      for name, col in (( "blue_cone", (200, 60, 20)), ("yellow_cone", (20, 200, 240))):
        x1, y1 = int(rng.uniform(200, 1000)) + border, int(rng.uniform(400, 600)) + border
        x2, y2 = x1 + 40, y1 + 60
        cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)
        objs.append({"classTitle": name, "geometryType": "rectangle", "points": {"exterior": [[x1, y1], [x2, y2]]}})
      cv2.imwrite(str(root / t / "img" / f"{i}.jpg"), img)
      (root / t / "ann" / f"{i}.jpg.json").write_text(json.dumps({"size": {"height": H, "width": W}, "objects": objs}))


def test_import_and_dataset(tmp_path):
  src = tmp_path / "fsoco"
  _make_sly(src)
  idx = import_fsoco(src, tmp_path / "out", border=20)
  assert len(idx["images"]) == 12
  assert sum(len(v) for v in idx["splits"].values()) == 12
  teams_per_split = {k: {idx["images"][i]["team"] for i in v} for k, v in idx["splits"].items()}
  for a in teams_per_split:
    for b in teams_per_split:
      if a != b:
        assert not (teams_per_split[a] & teams_per_split[b]), "team leakage across splits"
  split = next(k for k, v in idx["splits"].items() if v)
  ds = FsocoDataset(tmp_path / "out", split, augment=False)
  s = ds[0]
  assert s["img"].shape == (12, 128, 256) and s["big_img"].shape == (12, 128, 256)
  assert s["wide_pos"].sum() >= 1  # both cones land in the wide slot
  assert s["wide_heat"].shape == (4, 64, 128)
