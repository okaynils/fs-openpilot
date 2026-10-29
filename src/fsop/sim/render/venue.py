"""Venue realism for the Panda3D renderer: sky panoramas, event-ground asphalt, perimeter scenery,
people (incl. hi-vis marshals) and ground shadows.

Real FS venues are large open asphalt areas: light, patched, tyre-marked ground with painted markings,
and at the horizon barrier walls with banners, fences, grandstands, tents, vans and people. The model
only sees the camera, so these are what it has to learn to look past. Everything is randomized per
episode (no single venue is replicated), and enabled through ``SceneDR.venue``.

Sky panoramas: CC0 tonemapped equirectangular photos from Poly Haven (``fsop sim fetch-assets``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from panda3d.core import (
  CardMaker,
  Geom,
  GeomNode,
  GeomTriangles,
  GeomVertexData,
  GeomVertexFormat,
  GeomVertexWriter,
  NodePath,
  Texture,
  TextureStage,
  TransparencyAttrib,
)

PANORAMA_DIR = Path("artifacts/panoramas")
# open, daytime, flat surroundings (parking lots, airfields, roads, fields), CC0 (polyhaven.com)
PANORAMAS = (
  "abandoned_parking", "abandoned_tank_farm_01", "abandoned_tank_farm_03", "derelict_airfield_01",
  "derelict_airfield_02", "derelict_highway_midday", "derelict_highway_noon", "driving_school",
  "hanger_exterior_cloudy", "german_town_street", "future_parking", "glencairn_expressway", "crosswalk",
  "autumn_road", "abandoned_hopper_terminal_04", "countrytrax_midday", "farm_field_puresky",
  "autumn_field_puresky", "dirt_bike_track_01", "bloem_olive_house",
)
GROUND_TILE_M = 128.0


def fetch_panoramas(out: str | Path = PANORAMA_DIR, width: int = 4096) -> list[Path]:
  """Download the tonemapped JPGs from Poly Haven and store them at ``width`` x ``width/2``."""
  import json
  import urllib.request

  def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fs-openpilot (https://github.com/okaynils/fs-openpilot)"})
    return urllib.request.urlopen(req, timeout=120).read()

  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  paths = []
  for name in PANORAMAS:
    p = out / f"{name}.jpg"
    if not p.exists():
      meta = json.loads(get(f"https://api.polyhaven.com/files/{name}"))
      raw = get(meta["tonemapped"]["url"])
      img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
      img = cv2.resize(img, (width, width // 2), interpolation=cv2.INTER_AREA)
      cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    paths.append(p)
  return paths


def available_panoramas(root: str | Path = PANORAMA_DIR) -> list[str]:
  root = Path(root)
  return sorted(p.stem for p in root.glob("*.jpg")) if root.exists() else []


def _np_texture(rgb: np.ndarray, name: str, alpha: np.ndarray | None = None, repeat: bool = True) -> Texture:
  """float RGB [H,W,3] in 0..1 (row 0 = top) -> Panda texture."""
  h, w = rgb.shape[:2]
  img = np.clip(rgb[::-1], 0, 1)  # Panda's RAM images are bottom-up
  tex = Texture(name)
  if alpha is None:
    tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb8)
    tex.setRamImage((img[..., ::-1] * 255).astype(np.uint8).tobytes())  # BGR
  else:
    a = np.clip(alpha[::-1], 0, 1)[..., None]
    tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba8)
    tex.setRamImage((np.concatenate([img[..., ::-1], a], -1) * 255).astype(np.uint8).tobytes())  # BGRA
  mode = Texture.WM_repeat if repeat else Texture.WM_clamp
  tex.setWrapU(mode)
  tex.setWrapV(mode)
  tex.setMinfilter(Texture.FT_linear_mipmap_linear)
  tex.setMagfilter(Texture.FT_linear)
  tex.setAnisotropicDegree(8)
  return tex


# ---------------------------------------------------------------------------------------------- sky
def sky_dome(parent: NodePath, panorama: str | Path, radius: float = 450.0, rings: int = 24,
             segments: int = 48, yaw: float = 0.0) -> NodePath:
  """Inward-facing sphere with an equirectangular photo; recentred on the camera every frame."""
  fmt = GeomVertexFormat.getV3t2()
  vdata = GeomVertexData("sky", fmt, Geom.UHStatic)
  vw, tw = GeomVertexWriter(vdata, "vertex"), GeomVertexWriter(vdata, "texcoord")
  for i in range(rings + 1):
    el = np.pi / 2 - np.pi * i / rings  # +90 (zenith) .. -90
    for j in range(segments + 1):
      az = 2 * np.pi * j / segments
      vw.addData3(radius * np.cos(el) * np.cos(az + yaw), radius * np.cos(el) * np.sin(az + yaw), radius * np.sin(el))
      tw.addData2(1.0 - j / segments, 1.0 - i / rings)
  tris = GeomTriangles(Geom.UHStatic)
  for i in range(rings):
    for j in range(segments):
      a, b = i * (segments + 1) + j, (i + 1) * (segments + 1) + j
      tris.addVertices(a, b, a + 1)
      tris.addVertices(a + 1, b, b + 1)
  geom = Geom(vdata)
  geom.addPrimitive(tris)
  node = GeomNode("sky")
  node.addGeom(geom)
  dome = parent.attachNewNode(node)
  img = cv2.cvtColor(cv2.imread(str(panorama)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
  dome.setTexture(_np_texture(img, "sky", repeat=False), 1)
  dome.setTwoSided(True)
  dome.setLightOff(1)
  dome.setFogOff(1)
  dome.setDepthWrite(False)
  dome.setBin("background", 0)
  return dome


# ---------------------------------------------------------------------------------------------- ground
def venue_ground(rng: np.random.Generator, base: float, tint, size: int = 2048) -> np.ndarray:
  """Event-ground asphalt over ``GROUND_TILE_M`` metres: patches with seams, tyre marks, stains, cracks
  and painted markings. Float RGB [size,size,3]."""
  px = size / GROUND_TILE_M  # pixels per metre
  g = np.zeros((size, size), np.float32)
  for octave, amp in ((4, 0.5), (16, 0.3), (64, 0.15)):  # gentle low-frequency variation
    n = rng.normal(0, 1, (octave, octave)).astype(np.float32)
    g += amp * cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)
  g = base + 0.035 * g / max(np.abs(g).max(), 1e-6)
  # resurfaced patches: polygons a bit darker/lighter, with a dark tar seam
  for _ in range(int(rng.integers(3, 14))):
    c = rng.uniform(0, size, 2)
    w, h = rng.uniform(4, 40, 2) * px
    ang = rng.uniform(0, 180) if rng.random() < 0.3 else rng.choice([0.0, 90.0])
    box = cv2.boxPoints(((float(c[0]), float(c[1])), (float(w), float(h)), float(ang))).astype(np.int32)
    m = np.zeros_like(g)
    cv2.fillPoly(m, [box], 1.0)
    g += m * rng.uniform(-0.07, 0.05)
    cv2.polylines(g, [box], True, float(base - 0.12), max(1, int(0.06 * px)), cv2.LINE_AA)
  # stains (oil, wet patches): irregular dark blobs
  for _ in range(int(rng.integers(20, 90))):
    m = np.zeros_like(g)
    c = rng.uniform(0, size, 2)
    for _ in range(int(rng.integers(1, 5))):
      ax = rng.uniform(0.2, 2.0, 2) * px
      off = rng.normal(0, 0.5 * ax.mean(), 2)
      cv2.ellipse(m, (int(c[0] + off[0]), int(c[1] + off[1])), (int(ax[0]), int(ax[1])), float(rng.uniform(0, 180)),
                  0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 0.08 * px + 0.5)
    g -= m * rng.uniform(0.05, 0.22)
  # tyre marks: pairs of soft dark curves ~1.2 m apart
  for _ in range(int(rng.integers(15, 60))):
    p0, d = rng.uniform(0, size, 2), rng.uniform(0, 2 * np.pi)
    k = rng.normal(0, 0.02)  # curvature [1/m]
    pts = []
    for _ in np.arange(0, rng.uniform(8, 60), 0.5):
      d += k * 0.5
      p0 = p0 + 0.5 * px * np.array([np.cos(d), np.sin(d)])
      pts.append(p0.copy())
    pts = np.array(pts)
    nrm = np.array([-np.sin(d), np.cos(d)])
    m = np.zeros_like(g)
    for side in (-0.6, 0.6):
      cv2.polylines(m, [(pts + side * px * nrm).astype(np.int32)], False, 1.0, max(1, int(0.2 * px)), cv2.LINE_AA)
    g -= cv2.GaussianBlur(m, (0, 0), 1.0) * rng.uniform(0.03, 0.12)
  # cracks: thin dark random walks
  for _ in range(int(rng.integers(5, 40))):
    p = rng.uniform(0, size, 2)
    d = rng.uniform(0, 2 * np.pi)
    pts = [p.copy()]
    for _ in range(int(rng.integers(5, 40))):
      d += rng.normal(0, 0.5)
      p = p + rng.uniform(0.1, 0.4) * px * np.array([np.cos(d), np.sin(d)])
      pts.append(p.copy())
    cv2.polylines(g, [np.array(pts, np.int32)], False, float(base - 0.15), 1, cv2.LINE_AA)
  rgb = np.repeat(g[..., None], 3, -1) * np.asarray(tint, np.float32)
  # painted markings: long straight lines (white / yellow / red), parking grids; worn
  for _ in range(int(rng.integers(0, 10))):
    col = [(0.85, 0.85, 0.82), (0.88, 0.75, 0.2), (0.75, 0.2, 0.18)][int(rng.choice(3, p=[0.5, 0.25, 0.25]))]
    width = max(1, int(rng.uniform(0.08, 0.15) * px))
    m = np.zeros_like(g)
    p, d = rng.uniform(0, size, 2), rng.uniform(0, np.pi)
    if rng.random() < 0.3:  # parking grid
      for k in range(int(rng.integers(3, 12))):
        q = p + k * 2.6 * px * np.array([np.cos(d), np.sin(d)])
        e = q + 5.0 * px * np.array([-np.sin(d), np.cos(d)])
        cv2.line(m, tuple(q.astype(int)), tuple(e.astype(int)), 1.0, width, cv2.LINE_AA)
    else:
      e = p + rng.uniform(10, 120) * px * np.array([np.cos(d), np.sin(d)])
      cv2.line(m, tuple(p.astype(int)), tuple(e.astype(int)), 1.0, width, cv2.LINE_AA)
    wear = np.clip(rng.uniform(0.55, 0.95) + 0.25 * cv2.resize(rng.normal(0, 1, (64, 64)).astype(np.float32),
                                                                (size, size)), 0, 1)
    a = (m * wear)[..., None]
    rgb = rgb * (1 - a) + np.asarray(col, np.float32) * a
  return np.clip(rgb, 0, 1)


def grain_texture(rng: np.random.Generator, size: int = 256) -> Texture:
  """Fine aggregate grain, modulated over the large ground texture (~1-2 m tiles)."""
  n = rng.normal(0, 1, (size, size)).astype(np.float32)
  n = 0.6 * n + 0.8 * cv2.GaussianBlur(n, (0, 0), 2.0)
  g = 0.94 + 0.06 * np.clip(n / 2.5, -1, 1)  # modulate can only darken: centred just below 1
  return _np_texture(np.repeat(g[..., None], 3, -1), "grain")


# ---------------------------------------------------------------------------------------------- geometry helpers
def _box(parent: NodePath, center, size, yaw: float, color, name: str = "box", tex: Texture | None = None) -> NodePath:
  """Axis box on the ground (center x,y; size sx,sy,sz); optional texture on the long faces."""
  cm = CardMaker(name)
  box = parent.attachNewNode(name)
  sx, sy, sz = size
  for h, (w, d), textured in ((0, (sx, sy), True), (90, (sy, sx), False), (180, (sx, sy), True),
                              (270, (sy, sx), False)):
    cm.setFrame(-w / 2, w / 2, 0, sz)
    f = box.attachNewNode(cm.generate())
    f.setH(h)
    f.setY(f, -d / 2)
    if tex is not None and textured:
      f.setTexture(tex, 1)
  cm.setFrame(-sx / 2, sx / 2, -sy / 2, sy / 2)
  top = box.attachNewNode(cm.generate())
  top.setP(-90)
  top.setZ(sz)
  box.setPos(float(center[0]), float(center[1]), float(center[2]) if len(center) > 2 else 0.0)
  box.setH(float(np.rad2deg(yaw)))
  box.setColor(*color, 1)
  box.setTwoSided(True)
  return box


def banner_texture(rng: np.random.Generator, w: int = 512, h: int = 64) -> Texture:
  """Sponsor-banner-like strip: coloured panels with high-contrast blocks ("logos") and text-like bars."""
  img = np.zeros((h, w, 3), np.float32)
  x = 0
  while x < w:
    pw = int(rng.integers(60, 200))
    bg = rng.uniform(0, 1, 3) if rng.random() < 0.6 else np.full(3, rng.choice([0.05, 0.95]))
    img[:, x:x + pw] = bg
    fg = 1 - bg if rng.random() < 0.5 else rng.uniform(0, 1, 3)
    for _ in range(int(rng.integers(1, 6))):
      bx, by = x + int(rng.integers(0, max(pw - 20, 1))), int(rng.integers(5, h - 20))
      cv2.rectangle(img, (bx, by), (bx + int(rng.integers(8, 60)), by + int(rng.integers(4, 18))), fg.tolist(), -1)
    x += pw
  return _np_texture(img, "banner")


def fence_texture(rng: np.random.Generator, color) -> Texture:
  n = 128
  a = np.zeros((n, n), np.float32)
  step = int(rng.integers(6, 14))
  for i in range(0, n, step):
    cv2.line(a, (i, 0), (i, n), 1.0, 1)
    cv2.line(a, (0, i), (n, i), 1.0, 1)
  a = np.clip(a * rng.uniform(0.5, 0.9), 0, 1)
  if rng.random() < 0.5:  # privacy mesh / tarp
    a = np.maximum(a, rng.uniform(0.6, 0.95))
  rgb = np.ones((n, n, 3), np.float32) * np.asarray(color, np.float32)
  return _np_texture(rgb, "fence", alpha=a)


def person(parent: NodePath, x: float, y: float, yaw: float, rng: np.random.Generator, vest=None) -> NodePath:
  """Low-poly standing person (~1.75 m); ``vest`` colour for marshals (hi-vis)."""
  p = parent.attachNewNode("person")
  scale = rng.uniform(0.9, 1.08)
  legs = rng.choice([0.1, 0.15, 0.2, 0.35]) * np.ones(3) + rng.normal(0, 0.03, 3)
  torso = np.asarray(vest) if vest is not None else rng.uniform(0.05, 0.9, 3)
  skin = np.array([0.85, 0.65, 0.5]) * rng.uniform(0.45, 1.05)
  _box(p, (0, 0, 0), (0.22, 0.38, 0.85), 0, np.clip(legs, 0, 1), "legs")
  _box(p, (0, 0, 0.85), (0.26, 0.46, 0.6), 0, np.clip(torso, 0, 1), "torso")
  _box(p, (0, 0, 1.45), (0.2, 0.18, 0.25), 0, np.clip(skin, 0, 1), "head")
  p.setPos(x, y, 0)
  p.setH(float(np.rad2deg(yaw)))
  p.setScale(scale)
  return p


# ---------------------------------------------------------------------------------------------- shadows
def shadow_polys(cones_xy: np.ndarray, radius: float, height: float, sun_az: float, sun_el: float) -> list[np.ndarray]:
  """Ground shadow polygon per cone: base disc + the apex projected away from the sun."""
  d = -np.array([np.cos(sun_az), np.sin(sun_az)])  # shadows point away from the sun
  L = height / np.tan(max(sun_el, 0.15))
  n = np.array([-d[1], d[0]])
  out = []
  for c in cones_xy:
    ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    disc = c + radius * np.c_[np.cos(ang), np.sin(ang)]
    tip = c + d * L
    out.append(np.vstack([disc, c + n * radius * 0.9, tip, c - n * radius * 0.9]))
  return out


def ground_shadows(parent: NodePath, polys: list[np.ndarray], alpha: float, z: float = 0.003) -> NodePath:
  """Flat translucent dark polygons (convex hull triangulation) on the ground."""
  fmt = GeomVertexFormat.getV3c4()
  vdata = GeomVertexData("shadow", fmt, Geom.UHStatic)
  vw, cw = GeomVertexWriter(vdata, "vertex"), GeomVertexWriter(vdata, "color")
  tris = GeomTriangles(Geom.UHStatic)
  k = 0
  for poly in polys:
    hull = cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2)
    cx, cy = hull.mean(0)
    vw.addData3(float(cx), float(cy), z)
    cw.addData4(0, 0, 0, alpha)
    for p in hull:
      vw.addData3(float(p[0]), float(p[1]), z)
      cw.addData4(0, 0, 0, alpha)
    m = len(hull)
    for i in range(m):
      tris.addVertices(k, k + 1 + i, k + 1 + (i + 1) % m)
    k += m + 1
  geom = Geom(vdata)
  geom.addPrimitive(tris)
  node = GeomNode("shadows")
  node.addGeom(geom)
  np_ = parent.attachNewNode(node)
  np_.setTransparency(TransparencyAttrib.MAlpha)
  np_.setLightOff(1)
  np_.setTwoSided(True)
  np_.setDepthWrite(False)
  np_.setBin("fixed", 0)
  return np_


# ---------------------------------------------------------------------------------------------- scenery
def build_venue(parent: NodePath, track, rng: np.random.Generator) -> np.ndarray:
  """Perimeter (walls + banners, fences, grandstand, tents, vans, tyre stacks) and people.
  Returns the people's ground positions [N,2] (they cast shadows like cones)."""
  from shapely.geometry import LineString, Point

  root = parent.attachNewNode("venue")
  lo = track.centerline.min(0)
  hi = track.centerline.max(0)
  margin = rng.uniform(15, 50, 4)  # per side
  x0, y0, x1, y1 = lo[0] - margin[0], lo[1] - margin[1], hi[0] + margin[2], hi[1] + margin[3]
  sides = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
  line = LineString(track.centerline)
  half_w = float(max(track.width_left.max(), track.width_right.max()))
  shadows = []
  wall_color = np.full(3, rng.uniform(0.55, 0.85))
  fence_col = rng.choice([[0.1, 0.25, 0.12], [0.2, 0.2, 0.22], [0.08, 0.15, 0.3]]) * rng.uniform(0.8, 1.3)
  for (a, b) in sides:
    a, b = np.array(a), np.array(b)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / max(L, 1e-6)
    out = np.array([d[1], -d[0]])  # outward normal (counter-clockwise rectangle)
    yaw = float(np.arctan2(d[1], d[0]))
    s = 0.0
    while s < L:  # barrier wall segments, some carrying banners
      seg = float(min(rng.uniform(6, 20), L - s))
      c = a + d * (s + seg / 2)
      if rng.random() < 0.85 and line.distance(Point(c)) > half_w + 8.0:
        hgt = rng.uniform(0.8, 1.3)
        tex = banner_texture(rng) if rng.random() < 0.6 else None
        _box(root, (*c, 0), (seg, 0.4, hgt), yaw, wall_color * rng.uniform(0.9, 1.1), "wall", tex)
      s += seg
    if rng.random() < 0.7:  # fence behind the wall
      cm = CardMaker("fence")
      cm.setFrame(0, L, 0, rng.uniform(1.8, 2.6))
      f = root.attachNewNode(cm.generate())
      p = a + out * rng.uniform(1.0, 4.0)
      f.setPos(float(p[0]), float(p[1]), 0)
      f.setH(float(np.rad2deg(yaw)))
      tex = fence_texture(rng, fence_col)
      f.setTexture(tex, 1)
      f.setTexScale(TextureStage.getDefault(), L / 1.5, 1.0)
      f.setTransparency(TransparencyAttrib.MAlpha)
      f.setTwoSided(True)
    c = a + d * rng.uniform(0.2, 0.8) * L + out * rng.uniform(6, 15)
    if rng.random() < 0.3 and line.distance(Point(c)) > half_w + 15.0:  # grandstand with a crowd texture
      w = rng.uniform(15, 50)
      for k in range(int(rng.integers(3, 8))):
        crowd = rng.uniform(0, 1, (16, 128, 3)).astype(np.float32) * rng.uniform(0.3, 1.0)
        _box(root, (*(c + out * k * 0.9), k * 0.6), (w, 0.9, 0.6 + 0.3), yaw, (1, 1, 1), "stand",
             _np_texture(crowd, "crowd"))
    for _ in range(int(rng.integers(0, 4))):  # tents, vans, containers along the perimeter
      c = a + d * rng.uniform(0, L) + out * rng.uniform(1, 12)
      if line.distance(Point(c)) < half_w + 10.0:
        continue
      kind = rng.choice(["tent", "van", "container", "tyres"])
      if kind == "tent":
        sz = rng.uniform([3, 3, 2.2], [8, 6, 3.2])
        _box(root, (*c, 0), sz, yaw, (rng.uniform(0.75, 1.0),) * 3, "tent")
      elif kind == "van":
        col = (0.92, 0.92, 0.92) if rng.random() < 0.6 else rng.uniform(0.1, 0.9, 3)
        _box(root, (*c, 0), (5.2, 2.0, 2.2), yaw + rng.uniform(-0.3, 0.3), col, "van")
        _box(root, (*c, 1.3), (5.25, 2.05, 0.5), yaw, (0.08, 0.1, 0.12), "windows")
      elif kind == "container":
        _box(root, (*c, 0), (6.0, 2.4, 2.6), yaw, rng.uniform(0.15, 0.8, 3), "container")
      else:
        for i in range(int(rng.integers(2, 8))):
          _box(root, (*(c + d * i * 0.7), 0), (0.6, 0.6, rng.uniform(0.5, 1.2)), 0, (0.05, 0.05, 0.05), "tyres")
  # people: spectators near the perimeter, marshals (hi-vis) closer to the track
  hivis = [(0.85, 0.95, 0.1), (1.0, 0.55, 0.05), (0.95, 0.85, 0.15)]
  n_people = int(rng.integers(0, 35))
  for i in range(n_people):
    marshal = i < n_people // 4
    for _ in range(20):
      if marshal:
        p = rng.uniform(lo - 5, hi + 5)
        ok = half_w + 3.0 < line.distance(Point(p)) < half_w + 15.0
      else:
        side = sides[int(rng.integers(4))]
        a, b = np.array(side[0]), np.array(side[1])
        p = a + (b - a) * rng.uniform(0, 1) + rng.normal(0, 2.0, 2)
        ok = line.distance(Point(p)) > half_w + 5.0
      if ok:
        vest = hivis[int(rng.integers(len(hivis)))] if marshal or rng.random() < 0.1 else None
        person(root, float(p[0]), float(p[1]), float(rng.uniform(0, 2 * np.pi)), rng, vest)
        shadows.append(p)
        break
  return np.array(shadows).reshape(-1, 2)
