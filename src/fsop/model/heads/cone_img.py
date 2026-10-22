"""Image-space cone head (CenterNet) on the stride-4/8 trunk taps, one output set per camera slot.

Because both cameras are fused in the trunk, the head predicts separate maps for the road (medmodel)
and wide (sbigmodel) frames. Supervised by sim projections and by FSOCO (real images) - the shared
supervision that bridges sim and real.
Output resolution: 64x128 (stride 4 on the 512x256 model frame).
"""

from __future__ import annotations

import torch
from torch import nn

from fsop.data.targets import N_CLS

CAMS = ("road", "wide")


class ConeImageHead(nn.Module):
  def __init__(self, d: int = 96):
    super().__init__()
    self.lat8 = nn.Conv2d(128, d, 1)
    self.lat4 = nn.Conv2d(64, d, 1)
    self.lat16 = nn.Conv2d(256, d, 1)
    self.fuse = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU(),
                              nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU())
    self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                            nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU())
    self.out = nn.ModuleDict({c: nn.Conv2d(d, N_CLS + 4, 3, padding=1) for c in CAMS})
    for c in CAMS:
      nn.init.constant_(self.out[c].bias[:N_CLS], -2.19)

  def forward(self, s4: torch.Tensor, s8: torch.Tensor, s16: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    up = lambda x, ref: nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
    x = self.lat4(s4) + up(self.lat8(s8), s4) + up(self.lat16(s16), s4)
    x = self.up(self.fuse(x))
    res = {}
    for c in CAMS:
      o = self.out[c](x)
      res[c] = {"heat": o[:, :N_CLS], "wh": o[:, N_CLS:N_CLS + 2], "off": o[:, N_CLS + 2:]}
    return res
