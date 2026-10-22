"""FSOCO (Formula Student Objects in Context) import + dataset for the image-space cone head.

Supported layouts:
  * Supervisely export (official):  <root>/meta.json, <root>/<team>/{img,ann}/<file>(.json)
    objects: classTitle in {blue_cone, yellow_cone, orange_cone, large_orange_cone, unknown_cone},
    geometryType "rectangle" (points.exterior = [[x1,y1],[x2,y2]]) or "bitmap"/"polygon" (bbox derived)
  * YOLO export (fsoco tools sly2yolo): <root>/{images,labels}/<split?>/... with class ids
    YOLO_CLASSES below (check against your export's data.yaml).
FSOCO images carry a watermark border; ``border`` crops it (to verify: the devkit's
FSOCO_IMPORT_BORDER_THICKNESS). Splits are by team to avoid near-duplicate leakage.

Real images have unknown intrinsics. Each one is mapped into BOTH model-frame camera slots with an
assumed horizontal FOV (randomized as augmentation) and a horizon row: the wide slot shows the whole
image at the sbigmodel focal length, the road slot the 2x-zoomed centre (medmodel), exactly the
relation between the two openpilot cameras. The previous-frame slot repeats the frame (no ego motion).
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from fsop.data.targets import image_targets
from fsop.geometry.camera import CAMERA_MODELS
from fsop.geometry.yuv import rgb_to_model

SLY_CLASSES = {"blue_cone": 0, "yellow_cone": 1, "orange_cone": 2, "large_orange_cone": 3, "unknown_cone": -1}
YOLO_CLASSES = {0: 1, 1: 0, 2: 2, 3: 3, 4: -1}  # fsoco yolo ids (yellow=0, blue=1, orange, large_orange, unknown)
IMG_EXT = (".jpg", ".jpeg", ".png")


def _split_of(team: str, val: float = 0.1, test: float = 0.1) -> str:
  h = (zlib.crc32(team.encode()) % 1000) / 1000
  return "test" if h < test else "val" if h < test + val else "train"


def _sly_boxes(ann: dict) -> list[list[float]]:
  out = []
  for o in ann.get("objects", []):
    cls = SLY_CLASSES.get(o.get("classTitle"), None)
    if cls is None:
      continue
    g = o.get("geometryType")
    if g == "rectangle":
      (x1, y1), (x2, y2) = o["points"]["exterior"]
    elif g == "polygon":
      pts = np.array(o["points"]["exterior"], float)
      x1, y1 = pts.min(0)
      x2, y2 = pts.max(0)
    elif g == "bitmap":
      ox, oy = o["bitmap"]["origin"]
      import base64
      import io
      import zlib as _z

      from PIL import Image

      m = np.array(Image.open(io.BytesIO(_z.decompress(base64.b64decode(o["bitmap"]["data"])))))
      ys, xs = np.nonzero(m[..., -1] if m.ndim == 3 else m)
      if len(xs) == 0:
        continue
      x1, y1, x2, y2 = ox + xs.min(), oy + ys.min(), ox + xs.max(), oy + ys.max()
    else:
      continue
    out.append([float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2)), cls])
  return out


def import_fsoco(src: str | Path, out: str | Path, fmt: str = "auto", border: int = 140) -> dict:
  src, out = Path(src), Path(out)
  out.mkdir(parents=True, exist_ok=True)
  if fmt == "auto":
    fmt = "supervisely" if (src / "meta.json").exists() or any(src.glob("*/ann")) else "yolo"
  images = []
  if fmt == "supervisely":
    for team_dir in sorted(p for p in src.iterdir() if (p / "ann").is_dir()):
      for ann_path in sorted((team_dir / "ann").glob("*.json")):
        img_name = ann_path.name[:-5]
        img_path = team_dir / "img" / img_name
        if not img_path.exists():
          continue
        ann = json.loads(ann_path.read_text())
        images.append({"path": str(img_path.resolve()), "team": team_dir.name, "w": ann["size"]["width"],
                       "h": ann["size"]["height"], "boxes": _sly_boxes(ann)})
  else:
    for img_path in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT):
      lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
      if not lbl.exists():
        continue
      im = cv2.imread(str(img_path))
      h, w = im.shape[:2]
      boxes = []
      for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
          continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, YOLO_CLASSES.get(c, -1)])
      team = img_path.stem.split("_")[0]
      images.append({"path": str(img_path.resolve()), "team": team, "w": w, "h": h, "boxes": boxes})
      border = 0  # YOLO exports are already cropped
  splits = {"train": [], "val": [], "test": []}
  for i, im in enumerate(images):
    splits[_split_of(im["team"])].append(i)
  idx = {"format": fmt, "border": border, "images": images, "splits": splits}
  (out / "index.json").write_text(json.dumps(idx))
  return idx


def slot_homography(src_w: int, src_h: int, hfov: float, horizon: float, cam: str) -> np.ndarray:
  """3x3 mapping model-slot pixel -> source pixel for a level camera with the given HFOV."""
  f = (src_w / 2) / np.tan(hfov / 2)
  K_src = np.array([[f, 0, src_w / 2], [0, f, horizon], [0, 0, 1.0]])
  return K_src @ np.linalg.inv(CAMERA_MODELS[cam].K)


class FsocoDataset(Dataset):
  def __init__(self, root: str | Path, split: str = "train", augment: bool | None = None,
               hfov_deg: tuple[float, float] = (50.0, 75.0)):
    self.idx = json.loads((Path(root) / "index.json").read_text())
    self.items = [self.idx["images"][i] for i in self.idx["splits"][split]]
    self.border = self.idx.get("border", 0)
    self.augment = (split == "train") if augment is None else augment
    self.hfov = hfov_deg
    self.rng = np.random.default_rng()

  def __len__(self) -> int:
    return len(self.items)

  def load(self, i: int):
    it = self.items[i]
    img = cv2.cvtColor(cv2.imread(it["path"]), cv2.COLOR_BGR2RGB)
    b = self.border
    boxes = np.array(it["boxes"], np.float32).reshape(-1, 5)
    if b and img.shape[0] > 2 * b and img.shape[1] > 2 * b:
      img = img[b:-b, b:-b]
      boxes[:, :4] -= b
    return img, boxes

  def __getitem__(self, i: int) -> dict:
    img, boxes = self.load(i)
    h, w = img.shape[:2]
    rng = self.rng
    if self.augment and rng.random() < 0.5:  # mirror: blue <-> yellow
      img = img[:, ::-1].copy()
      boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
      cls = boxes[:, 4].copy()
      boxes[cls == 0, 4], boxes[cls == 1, 4] = 1, 0
    if self.augment:
      img = self._photometric(img, rng)
    hfov = np.deg2rad(rng.uniform(*self.hfov) if self.augment else np.mean(self.hfov))
    horizon = h * (rng.uniform(0.35, 0.55) if self.augment else 0.45)
    s: dict = {}
    for cam, key in (("road", "img"), ("wide", "big_img")):
      Hm = slot_homography(w, h, hfov, horizon, cam)
      slot = cv2.warpPerspective(img, Hm, (512, 256), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
      yuv = rgb_to_model(slot)
      s[key] = np.concatenate([yuv, yuv], axis=0)
      Hinv = np.linalg.inv(Hm)
      if len(boxes):
        corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], 1).reshape(-1, 2)
        p = np.c_[corners, np.ones(len(corners))] @ Hinv.T
        p = (p[:, :2] / p[:, 2:]).reshape(-1, 4, 2)
        bx = np.c_[p[..., 0].min(1), p[..., 1].min(1), p[..., 0].max(1), p[..., 1].max(1)]
        key_pt = np.c_[(bx[:, 0] + bx[:, 2]) / 2, bx[:, 3] - 1e-3]
        valid = (key_pt[:, 0] >= 0) & (key_pt[:, 0] < 512) & (key_pt[:, 1] >= 0) & (key_pt[:, 1] < 256)
        heat, wh, off, pos = image_targets(key_pt, bx, boxes[:, 4].astype(int), valid)
      else:
        heat, wh, off, pos = image_targets(np.zeros((0, 2)), np.zeros((0, 4)), np.zeros(0, int), np.zeros(0, bool))
      s.update({f"{cam}_heat": heat, f"{cam}_wh": wh, f"{cam}_off": off, f"{cam}_pos": pos,
                f"{cam}_valid": np.float32(1.0)})
    return s

  @staticmethod
  def _photometric(img: np.ndarray, rng) -> np.ndarray:
    x = img.astype(np.float32)
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
    gray = x.mean(-1, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.7, 1.3)
    if rng.random() < 0.3:
      x = cv2.GaussianBlur(x, (3, 3), 0)
    x = x + rng.normal(0, rng.uniform(0, 4), x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)
