"""BEV cone head: learned lifting from the fused-camera trunk features to an ego-frame grid.

The trunk concatenates both cameras channel-wise, so per-camera geometric lifting (IPM / LSS) does not
apply. Instead BEV queries cross-attend to image tokens (s16 [256,8,16] + final [1024,4,8]) with
learned positional embeddings - the model-frame camera geometry is fixed, so position embeddings can
absorb it. Output: CenterNet heatmap per cone class + sub-cell offsets on the GRID (64x64, 0.75 m).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from fsop.data.targets import GRID, N_CLS


class ConeBEVHead(nn.Module):
  def __init__(self, d: int = 128, q: int = 16, layers: int = 2, heads: int = 4, out_hw: tuple[int, int] | None = None):
    super().__init__()
    self.q = q
    self.out_hw = out_hw or (GRID.H, GRID.W)
    self.proj16 = nn.Sequential(nn.Conv2d(256, d, 1), nn.GELU())
    self.proj32 = nn.Sequential(nn.Conv2d(1024, d, 1), nn.GELU())
    self.pos16 = nn.Parameter(torch.randn(1, 8 * 16, d) * 0.02)
    self.pos32 = nn.Parameter(torch.randn(1, 4 * 8, d) * 0.02)
    self.query = nn.Parameter(torch.randn(1, q * q, d) * 0.02)
    layer = nn.TransformerDecoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.0, batch_first=True, norm_first=True)
    self.decoder = nn.TransformerDecoder(layer, layers)
    ups = []
    ch = d
    n_up = int(round(math.log2(self.out_hw[0] / q)))
    for _ in range(n_up):
      nxt = ch // 2 if ch > 32 else ch
      ups += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(ch, nxt, 3, padding=1),
              nn.GroupNorm(8, nxt), nn.GELU()]
      ch = nxt
    self.up = nn.Sequential(*ups)
    self.heat = nn.Conv2d(ch, N_CLS, 3, padding=1)
    self.off = nn.Conv2d(ch, 2, 3, padding=1)
    nn.init.constant_(self.heat.bias, -2.19)

  def forward(self, s16: torch.Tensor, final: torch.Tensor) -> dict[str, torch.Tensor]:
    B = s16.shape[0]
    t16 = self.proj16(s16).flatten(2).transpose(1, 2) + self.pos16
    t32 = self.proj32(final).flatten(2).transpose(1, 2) + self.pos32
    mem = torch.cat([t16, t32], 1)
    x = self.decoder(self.query.expand(B, -1, -1), mem)
    x = x.transpose(1, 2).reshape(B, -1, self.q, self.q)
    x = self.up(x)
    if x.shape[-2:] != self.out_hw:
      x = nn.functional.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
    return {"heat": self.heat(x), "off": self.off(x)}
