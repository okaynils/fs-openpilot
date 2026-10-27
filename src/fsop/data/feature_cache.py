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


def _done(ep: Episode, tag: str, hidden_only: bool) -> bool:
  return (ep.cache_dir(tag) / ("hidden.npy" if hidden_only else "s16.npy")).exists()


@torch.no_grad()
def cache_episode(model, ep: Episode, tag: str, device, batch: int = 32, overwrite: bool = False,
                  hidden_only: bool = False, frames: tuple[np.ndarray, np.ndarray] | None = None) -> Path:
  """``hidden_only``: just the 512-d history features (all stage 2 needs; ~1 KB instead of ~130 KB / frame).
  ``frames``: pre-loaded (road, wide) arrays."""
  d = ep.cache_dir(tag)
  if _done(ep, tag, hidden_only) and not overwrite:
    return d
  d.mkdir(exist_ok=True)
  T = ep.T
  hidden = np.zeros((T, 512), np.float16)
  final = None if hidden_only else np.zeros((T, 1024, 4, 8), np.float16)
  s16 = None if hidden_only else np.zeros((T, 256, 8, 16), np.float16)
  road, wide = frames or (ep.frames["road"][:], ep.frames["wide"][:])
  for a in range(0, T, batch):
    ks = np.arange(a, min(a + batch, T))
    k0 = np.maximum(ks - 1, 0)
    img = torch.from_numpy(np.concatenate([road[k0], road[ks]], axis=1)).to(device).float()
    big = torch.from_numpy(np.concatenate([wide[k0], wide[ks]], axis=1)).to(device).float()
    v = model.encode(img, big)
    hidden[ks] = v.hidden.cpu().numpy().astype(np.float16)
    if hidden_only:
      continue
    final[ks] = v.final.cpu().numpy().astype(np.float16)
    s16[ks] = v.s16.cpu().numpy().astype(np.float16)
  np.save(d / "hidden.npy", hidden)
  if not hidden_only:
    np.save(d / "final.npy", final)
    np.save(d / "s16.npy", s16)
  return d


def build_cache(dataset: str | Path, ckpt: str | Path | None = None, tag: str | None = None, batch: int = 32,
                hidden_only: bool = False, prefetch: int = 16) -> str:
  """Frames are read by ``prefetch`` threads ahead of the GPU (frame reads dominate on network storage)."""
  from concurrent.futures import ThreadPoolExecutor

  device = pick_device()
  model = build_base(ckpt).to(device).eval()
  tag = tag or ("op_b361e952" if ckpt is None else Path(ckpt).parent.parent.name + "_" + Path(ckpt).stem)
  eps = [Episode(Path(dataset) / e["id"]) for e in load_index(dataset)]
  eps = [ep for ep in eps if not _done(ep, tag, hidden_only)]

  def load(ep: Episode):
    return ep.frames["road"][:], ep.frames["wide"][:]

  with ThreadPoolExecutor(max(prefetch, 1)) as ex:
    futs = [ex.submit(load, ep) for ep in eps[:prefetch]]
    for i, ep in enumerate(tqdm(eps, desc="cache")):
      frames = futs[i].result()
      futs[i] = None  # release the frames once used
      if i + prefetch < len(eps):
        futs.append(ex.submit(load, eps[i + prefetch]))
      cache_episode(model, ep, tag, device, batch, hidden_only=hidden_only, frames=frames)
  return tag
