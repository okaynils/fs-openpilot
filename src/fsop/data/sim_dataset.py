"""Sim episodes -> training samples.

Episodes are recorded on the 5 Hz grid (``record_every = 4``), so for sample k:
  img     = road[k-1] ++ road[k]      (t-4, t at 20 Hz)        big_img likewise from wide
  history = hidden[k-24 .. k-1]       (features_buffer; zero padded at episode start, like modeld)
  desire  = desire[k-24 .. k]         ([25,8], already max-pooled per 5 Hz slot)
  plan    = counterfactual expert plan at k, converted to openpilot's frame
  BEV / image cone targets from episode cones, pose and camera mount

Two modes: ``cached`` (stage 1: trunk frozen, cached hidden/final/s16 features, no images) and
``images`` (stage 2: raw frames for trunk fine-tuning; history still from the cache).
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from fsop.data.targets import (
  GRID,
  bev_targets,
  cone_boxes_image,
  frustum_mask,
  image_targets,
  visible_cones,
  world_to_ego,
)
from fsop.geometry.camera import CameraMount
from fsop.geometry.op_frames import ego_to_op_plan
from fsop.model.temporal import N_FEAT
from fsop.track.schema import CONE_CLASSES, ConeColor

BIG = CONE_CLASSES.index(ConeColor.BIG_ORANGE)


class Episode:
  def __init__(self, path: str | Path):
    self.path = Path(path)
    self.meta = json.loads((self.path / "meta.json").read_text())
    self.T = int(self.meta["frames"])
    self.states = np.load(self.path / "states.npy")
    self.plan = np.load(self.path / "plan.npy")
    self.desire = np.load(self.path / "desire.npy")
    c = np.load(self.path / "cones.npy")
    self.cones_xy, self.cones_cls = c[:, :2], c[:, 2].astype(np.int64)
    self.knock_times = np.full(len(c), np.inf)
    events_path = self.path / "events.jsonl"
    if events_path.exists():
      for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("kind") == "doo" and "cone" in event:
          idx = int(event["cone"])
          self.knock_times[idx] = min(self.knock_times[idx], float(event["t"]))
    m = self.meta["mount"]
    self.T_ego_cam = CameraMount(**m).T_ego_cam()
    self.calibrated_pose = CameraMount(**self.meta.get("calibrated_mount", m)).T_ego_cam()
    self._mask = None
    pose_path = self.path / "camera_poses.npy"
    self.camera_poses = np.load(pose_path) if pose_path.exists() else None
    from fsop.sim.vehicle import VehicleParams
    self.vehicle = VehicleParams(**self.meta.get("vehicle", {}))
    mask_path = self.path / "sensor_masks.npz"
    self.sensor_masks = dict(np.load(mask_path)) if mask_path.exists() else {}

  def camera_pose(self, k: int) -> np.ndarray:
    return self.T_ego_cam if self.camera_poses is None else self.camera_poses[k]

  def observable(self, points: np.ndarray, k: int, cam: str = "wide") -> np.ndarray:
    from fsop.geometry.camera import CAMERA_MODELS, project_world
    from fsop.geometry.mounting import body_occluded

    uv, _, valid = project_world(points, np.eye(4), self.camera_pose(k), CAMERA_MODELS[cam])
    pixels = np.floor(uv).astype(int)
    if cam in self.sensor_masks:
      mask = self.sensor_masks[cam]
      valid &= mask[pixels[:, 1].clip(0, mask.shape[0]-1), pixels[:, 0].clip(0, mask.shape[1]-1)]
    return valid & ~body_occluded(points, self.camera_pose(k), self.vehicle)

  def training_mask(self, k: int) -> np.ndarray:
    if not self.sensor_masks and self.vehicle.body_height <= 0:
      return frustum_mask(GRID, self.camera_pose(k))
    xy = GRID.cell_centers().reshape(-1, 2)
    points = np.c_[xy, np.zeros(len(xy))]
    return (self.observable(points, k).reshape(GRID.H, GRID.W)
            * frustum_mask(GRID, self.camera_pose(k))).astype(np.float32)

  @cached_property
  def frames(self):
    return zarr.open_group(str(self.meta.get("frames_source", self.path / "frames.zarr")), mode="r")

  def frame_pair(self, cam: str, k: int) -> np.ndarray:
    a = self.frames[cam]
    k0 = max(k - 1, 0)
    return np.concatenate([a[k0], a[k]], axis=0)

  def pose(self, k: int) -> tuple[float, float, float]:
    s = self.states[k]
    return float(s[1]), float(s[2]), float(s[3])

  def cones_ego(self, k: int, visible_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    x, y, yaw = self.pose(k)
    e = world_to_ego(self.cones_xy, x, y, yaw)
    near = (np.linalg.norm(e, axis=1) < GRID.max_range + 5) & (self.knock_times > self.states[k, 0])
    e, cls = e[near], self.cones_cls[near]
    if visible_only:
      vis = visible_cones(e, self.camera_pose(k))
      from fsop.geometry.mounting import body_occluded
      vis &= ~body_occluded(np.c_[e, np.full(len(e), 0.02)], self.camera_pose(k), self.vehicle)
      e, cls = e[vis], cls[vis]
    return e, cls

  @property
  def bev_mask(self) -> np.ndarray:
    if self._mask is None:
      self._mask = frustum_mask(GRID, self.T_ego_cam)
    return self._mask

  def cache_dir(self, tag: str) -> Path:
    return self.path / f"cache_{tag}"


def load_index(root: str | Path) -> list[dict]:
  return json.loads((Path(root) / "index.json").read_text())["episodes"]


def split_episodes(root: str | Path, split: str, val_frac: float = 0.1, test_frac: float = 0.1) -> list[str]:
  """Split by *track* (held-out tracks), deterministic. Fixed layouts (skidpad/accel) split by episode."""
  out = []
  for e in load_index(root):
    key = e["track_id"] if e["discipline"] in ("trackdrive", "autocross") else e["id"]
    h = (abs(hash_str(key)) % 1000) / 1000
    s = "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"
    s = e.get("split", s)
    if s == split or split == "all":
      out.append(e["id"])
  return out


def hash_str(s: str) -> int:
  import zlib

  return zlib.crc32(s.encode())


class SimDataset(Dataset):
  def __init__(self, root: str | Path, split: str = "train", mode: str = "cached", cache_tag: str = "op_b361e952",
               min_k: int = 1, image_targets_on: bool = True, episodes: list[str] | None = None,
               context_version: int = 1):
    self.root = Path(root)
    ids = episodes if episodes is not None else split_episodes(root, split)
    self.eps = [Episode(self.root / i) for i in ids]
    self.mode = mode
    self.tag = cache_tag
    self.context_version = context_version
    self.image_targets_on = image_targets_on
    self.index = [(ei, k) for ei, ep in enumerate(self.eps) for k in range(min_k, ep.T)]
    self._cache: dict[int, dict] = {}

  def __len__(self) -> int:
    return len(self.index)

  def _feats(self, ei: int) -> dict:
    if ei not in self._cache:
      d = self.eps[ei].cache_dir(self.tag)
      self._cache[ei] = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in ("hidden", "final", "s16")
                         if (d / f"{n}.npy").exists()}
    return self._cache[ei]

  def history(self, ei: int, k: int) -> np.ndarray:
    hid = self._feats(ei)["hidden"]
    idx = np.arange(k - N_FEAT, k)
    out = np.zeros((N_FEAT, 512), np.float32)
    ok = idx >= 0
    out[ok] = hid[idx[ok]]
    return out

  def __getitem__(self, i: int) -> dict:
    ei, k = self.index[i]
    ep = self.eps[ei]
    s: dict = {"ep": ei, "k": k}
    # desire [25,8]
    idx = np.arange(k - N_FEAT, k + 1)
    des = np.zeros((N_FEAT + 1, 8), np.float32)
    des[idx >= 0] = ep.desire[idx[idx >= 0]]
    s["desire"] = des
    from fsop.model.conditioning import geometry_context
    s["geometry"] = geometry_context(CameraMount(**ep.meta.get("calibrated_mount", ep.meta["mount"])),
                                     ep.vehicle, self.context_version)
    s["traffic"] = np.array([1.0, 0.0], np.float32)
    s["plan"] = ego_to_op_plan(ep.plan[k]).astype(np.float32)
    s["speed"] = np.float32(ep.states[k, 4])
    s["history"] = self.history(ei, k)
    f = self._feats(ei)
    if self.mode == "cached":
      s["hidden"] = np.asarray(f["hidden"][k], np.float32)
      s["final"] = np.asarray(f["final"][k], np.float32)
      s["s16"] = np.asarray(f["s16"][k], np.float32)
    else:
      s["img"] = ep.frame_pair("road", k)
      s["big_img"] = ep.frame_pair("wide", k)
    cones, cls = ep.cones_ego(k)
    heat, off, pos, mask = bev_targets(cones, cls, ep.training_mask(k))
    s.update(bev_heat=heat, bev_off=off, bev_pos=pos, bev_mask=mask)
    if self.mode != "cached" and self.image_targets_on:
      for cam in ("road", "wide"):
        key, box, valid = cone_boxes_image(cones, cls == BIG, ep.camera_pose(k), cam)
        if ep.sensor_masks or ep.vehicle.body_height > 0:
          valid &= ep.observable(np.c_[cones, np.zeros(len(cones))], k, cam)
        h, wh, o, p = image_targets(key, box, cls, valid)
        s.update({f"{cam}_heat": h, f"{cam}_wh": wh, f"{cam}_off": o, f"{cam}_pos": p})
    return s


def collate(batch: list[dict]) -> dict:
  out = {}
  for k in batch[0]:
    v = [b[k] for b in batch]
    out[k] = torch.from_numpy(np.stack(v)) if isinstance(v[0], np.ndarray | np.generic) else torch.tensor(v)
  return out
