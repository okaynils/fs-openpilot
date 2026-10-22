"""openpilot input-queue semantics (port of selfdrive/modeld/compile_modeld.py, MIT).

At 20 Hz with frame_skip = 4:
  img      = frames (t-4, t) stacked -> [12,128,256]  (older frame in channels 0..5)
  features = hidden(t-96), hidden(t-92), ..., hidden(t-4)   (24 slots, 5 Hz; graph appends hidden(t))
  desire   = 100-step pulse queue max-pooled in groups of 4 -> [25,8]
The policy graph only reads the last 9 history slots (current + 8 past = 1.6 s).
"""

from __future__ import annotations

from collections import deque

import numpy as np

FRAME_SKIP = 4
N_FEAT = 24
DESIRE_Q = 100


class TemporalState:
  """Closed-loop state at the model rate (20 Hz), mirroring modeld."""

  def __init__(self, cams: tuple[str, ...] = ("road", "wide"), frame_skip: int = FRAME_SKIP):
    self.fs = frame_skip
    self.img_q = {c: deque(maxlen=frame_skip + 1) for c in cams}
    self.feat_q = deque([np.zeros(512, np.float32)] * (frame_skip * N_FEAT), maxlen=frame_skip * N_FEAT)
    self.desire_q = deque([np.zeros(8, np.float32)] * DESIRE_Q, maxlen=DESIRE_Q)
    self.prev_desire = np.zeros(8, np.float32)

  def push_frames(self, frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for c, f in frames.items():
      q = self.img_q[c]
      if not q:
        q.extend([f] * (self.fs + 1))
      else:
        q.append(f)
      out[c] = np.concatenate([q[0], q[-1]], axis=0)
    return out

  def push_desire(self, desire: np.ndarray) -> np.ndarray:
    pulse = np.where(desire - self.prev_desire > 0.99, desire, 0).astype(np.float32)
    self.prev_desire = desire.astype(np.float32)
    self.desire_q.append(pulse)
    d = np.stack(self.desire_q).reshape(-1, self.fs, 8).max(1)
    return d  # [25,8]

  def features(self) -> np.ndarray:
    f = np.stack(self.feat_q)  # [96,512]
    return f[:: self.fs]  # [24,512]

  def push_hidden(self, hidden: np.ndarray) -> None:
    self.feat_q.append(hidden.astype(np.float32))


def history_indices(k: int, n: int = N_FEAT) -> np.ndarray:
  """For data recorded on the 5 Hz grid: indices of the 24 past hidden states for sample k (-1 = pad)."""
  idx = np.arange(k - n, k)
  return np.where(idx >= 0, idx, -1)
