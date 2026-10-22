"""Cache frozen-trunk features per frame: hidden [T,512], final [T,1024,4,8], s16 [T,256,8,16] (fp16).

Stage-1 training (policy + BEV head) then never touches images. ~130 KB / frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fsop.data.sim_dataset import Episode, load_index
from fsop.model.build import build_base
from fsop.train.device import pick_device


@torch.no_grad()
def cache_episode(model, ep: Episode, tag: str, device, batch: int = 32, overwrite: bool = False) -> Path:
  d = ep.cache_dir(tag)
  if (d / "s16.npy").exists() and not overwrite:
    return d
  d.mkdir(exist_ok=True)
  T = ep.T
  hidden = np.zeros((T, 512), np.float16)
  final = np.zeros((T, 1024, 4, 8), np.float16)
  s16 = np.zeros((T, 256, 8, 16), np.float16)
  road, wide = ep.frames["road"][:], ep.frames["wide"][:]
  for a in range(0, T, batch):
    ks = np.arange(a, min(a + batch, T))
    k0 = np.maximum(ks - 1, 0)
    img = torch.from_numpy(np.concatenate([road[k0], road[ks]], axis=1)).to(device).float()
    big = torch.from_numpy(np.concatenate([wide[k0], wide[ks]], axis=1)).to(device).float()
    v = model.encode(img, big)
    hidden[ks] = v.hidden.cpu().numpy().astype(np.float16)
    final[ks] = v.final.cpu().numpy().astype(np.float16)
    s16[ks] = v.s16.cpu().numpy().astype(np.float16)
  np.save(d / "hidden.npy", hidden)
  np.save(d / "final.npy", final)
  np.save(d / "s16.npy", s16)
  return d


def build_cache(dataset: str | Path, ckpt: str | Path | None = None, tag: str | None = None, batch: int = 32) -> str:
  device = pick_device()
  model = build_base(ckpt).to(device).eval()
  tag = tag or ("op_b361e952" if ckpt is None else Path(ckpt).parent.parent.name + "_" + Path(ckpt).stem)
  for e in tqdm(load_index(dataset), desc="cache"):
    cache_episode(model, Episode(Path(dataset) / e["id"]), tag, device, batch)
  return tag
