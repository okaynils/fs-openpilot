"""YOLO baseline on FSOCO (optional extra: `uv sync --extra yolo`; Ultralytics is AGPL-3.0).

Trains on the same team-level split as the image head (writes an ultralytics data.yaml with symlinked
images and YOLO labels) and reports the same COCO metrics on the model-frame images.
"""

from __future__ import annotations

import json
from pathlib import Path

NAMES = ["blue", "yellow", "orange", "big_orange"]


def export_yolo(fsoco_root: str | Path, out: str | Path) -> Path:
  idx = json.loads((Path(fsoco_root) / "index.json").read_text())
  out = Path(out)
  border = idx.get("border", 0)
  import cv2

  for split, ids in idx["splits"].items():
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in ids:
      im = idx["images"][i]
      src = Path(im["path"])
      dst = out / "images" / split / f"{i:06d}{src.suffix}"
      if not dst.exists():
        img = cv2.imread(str(src))
        if border:
          img = img[border:-border, border:-border]
        cv2.imwrite(str(dst), img)
      w, h = im["w"] - 2 * border, im["h"] - 2 * border
      lines = []
      for x1, y1, x2, y2, c in im["boxes"]:
        if c < 0:
          continue
        x1, y1, x2, y2 = x1 - border, y1 - border, x2 - border, y2 - border
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
      (out / "labels" / split / f"{i:06d}.txt").write_text("\n".join(lines))
  y = out / "data.yaml"
  y.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {NAMES}\n")
  return y


def train_yolo(fsoco_root: str | Path, out: str | Path = "runs/yolo", model: str = "yolo11n.pt", epochs: int = 50,
               imgsz: int = 1280, device: str = "mps") -> Path:
  from ultralytics import YOLO

  data = export_yolo(fsoco_root, Path(out) / "data")
  m = YOLO(model)
  m.train(data=str(data), epochs=epochs, imgsz=imgsz, device=device, project=str(out), name="train")
  return Path(out) / "train" / "weights" / "best.pt"
