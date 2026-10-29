"""Procedural cone meshes (FS Driverless cones) and ground textures for Panda3D."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from panda3d.core import (
  Geom,
  GeomNode,
  GeomTriangles,
  GeomVertexData,
  GeomVertexFormat,
  GeomVertexWriter,
  Texture,
)

from fsop.track.rules import RULES
from fsop.track.schema import ConeColor

# body colour, stripe colour (FS cones: blue/white, yellow/black, orange/white)
CONE_COLORS = {
  ConeColor.BLUE: ((0.05, 0.22, 0.75), (0.95, 0.95, 0.95)),
  ConeColor.YELLOW: ((0.98, 0.80, 0.05), (0.06, 0.06, 0.06)),
  ConeColor.ORANGE: ((1.00, 0.42, 0.03), (0.95, 0.95, 0.95)),
  ConeColor.BIG_ORANGE: ((1.00, 0.38, 0.02), (0.95, 0.95, 0.95)),
  ConeColor.UNKNOWN: ((0.55, 0.55, 0.55), (0.9, 0.9, 0.9)),
}
# stripe bands as fractions of body height
STRIPES = {
  ConeColor.BLUE: [(0.45, 0.65)],
  ConeColor.YELLOW: [(0.45, 0.65)],
  ConeColor.ORANGE: [(0.30, 0.45), (0.60, 0.75)],
  ConeColor.BIG_ORANGE: [(0.30, 0.42), (0.60, 0.72)],
  ConeColor.UNKNOWN: [],
}


def cone_geom(color: ConeColor, segments: int = 20, fade: float = 0.0, lettering: bool = False) -> GeomNode:
  """``fade``: mix the body colour towards a pale, sun-bleached tone; ``lettering``: white marks in the band."""
  spec = RULES.large_cone if color == ConeColor.BIG_ORANGE else RULES.small_cone
  body, stripe = CONE_COLORS[color]
  body = tuple((1 - fade) * np.array(body) + fade * np.array((0.92, 0.9, 0.78)))
  bands = STRIPES[color]
  fmt = GeomVertexFormat.getV3n3c4()
  vdata = GeomVertexData("cone", fmt, Geom.UHStatic)
  vw, nw, cw = GeomVertexWriter(vdata, "vertex"), GeomVertexWriter(vdata, "normal"), GeomVertexWriter(vdata, "color")
  tris = GeomTriangles(Geom.UHStatic)
  idx = 0

  def quad(p0, p1, p2, p3, n, c):
    nonlocal idx
    for p in (p0, p1, p2, p3):
      vw.addData3(*p)
      nw.addData3(*n)
      cw.addData4(*c, 1.0)
    tris.addVertices(idx, idx + 1, idx + 2)
    tris.addVertices(idx, idx + 2, idx + 3)
    idx += 4

  # base plate (square, 2 cm high)
  b = spec.base / 2
  h0 = 0.02
  base_c = tuple(0.8 * np.array(body))
  quad((-b, -b, h0), (b, -b, h0), (b, b, h0), (-b, b, h0), (0, 0, 1), base_c)
  for (x0, y0), (x1, y1), n in (((-b, -b), (b, -b), (0, -1, 0)), ((b, -b), (b, b), (1, 0, 0)),
                                ((b, b), (-b, b), (0, 1, 0)), ((-b, b), (-b, -b), (-1, 0, 0))):
    quad((x0, y0, 0), (x1, y1, 0), (x1, y1, h0), (x0, y0, h0), n, base_c)
  # body: truncated cone split into colour bands
  H = spec.height - h0
  cuts = sorted({0.0, 1.0, *[v for band in bands for v in band]})
  slope = (spec.radius_bottom - spec.radius_top) / H
  for z0, z1 in zip(cuts[:-1], cuts[1:], strict=True):
    in_stripe = any(lo <= z0 < hi for lo, hi in bands)
    c0 = stripe if in_stripe else body
    r0 = spec.radius_bottom + (spec.radius_top - spec.radius_bottom) * z0
    r1 = spec.radius_bottom + (spec.radius_top - spec.radius_bottom) * z1
    for k in range(segments):
      a0, a1 = 2 * np.pi * k / segments, 2 * np.pi * (k + 1) / segments
      am = (a0 + a1) / 2
      n = (np.cos(am), np.sin(am), slope)
      c = (0.9, 0.9, 0.88) if lettering and in_stripe and k % 5 in (1, 2) else c0
      quad((r0 * np.cos(a0), r0 * np.sin(a0), h0 + z0 * H), (r0 * np.cos(a1), r0 * np.sin(a1), h0 + z0 * H),
           (r1 * np.cos(a1), r1 * np.sin(a1), h0 + z1 * H), (r1 * np.cos(a0), r1 * np.sin(a0), h0 + z1 * H), n, c)
  # top cap
  rt = spec.radius_top
  for k in range(segments):
    a0, a1 = 2 * np.pi * k / segments, 2 * np.pi * (k + 1) / segments
    quad((0, 0, spec.height), (rt * np.cos(a0), rt * np.sin(a0), spec.height),
         (rt * np.cos(a1), rt * np.sin(a1), spec.height), (0, 0, spec.height), (0, 0, 1), body)
  geom = Geom(vdata)
  geom.addPrimitive(tris)
  node = GeomNode(f"cone_{color.value}")
  node.addGeom(geom)
  return node


def metadrive_texture(name: str) -> Path | None:
  """Optional: reuse MetaDrive's asset textures (asphalt, grass, skybox) when the sim extra is installed."""
  try:
    import metadrive

    root = Path(metadrive.__file__).parent / "assets" / "textures"
  except ImportError:
    return None
  p = {"asphalt": root / "asphalt" / "diff_2k.png", "grass": root / "grass1" / "GroundGrassGreen002_COL_1K.jpg",
       "skybox": root / "skybox.jpg"}.get(name)
  return p if p is not None and p.exists() else None


def procedural_asphalt(rng: np.random.Generator, size: int = 512, base: float = 0.33,
                       contrast: float = 0.06) -> Texture:
  """Multi-octave value noise asphalt with optional painted lines; returned as a tileable texture."""
  import cv2

  img = np.zeros((size, size), np.float32)
  for octave, amp in ((8, 0.5), (32, 0.3), (128, 0.15), (512, 0.05)):
    g = rng.normal(0, 1, (octave, octave)).astype(np.float32)
    img += amp * cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC if octave < size else cv2.INTER_NEAREST)
  img = base + contrast * img / max(np.abs(img).max(), 1e-6)
  tint = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.03, 3)
  rgb = np.clip(img[..., None] * tint, 0, 1)
  tex = Texture("asphalt")
  tex.setup2dTexture(size, size, Texture.T_unsigned_byte, Texture.F_rgb8)
  tex.setRamImage((rgb[..., ::-1] * 255).astype(np.uint8).tobytes())
  tex.setWrapU(Texture.WM_repeat)
  tex.setWrapV(Texture.WM_repeat)
  tex.setMinfilter(Texture.FT_linear_mipmap_linear)
  tex.setAnisotropicDegree(4)
  return tex
