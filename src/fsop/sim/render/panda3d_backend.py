"""Offscreen Panda3D renderer producing openpilot model-frame images directly.

Each camera renders into its own 512x256 buffer with the exact model-frame intrinsics (fl, cy), so no
warp is needed and pixel <-> 3D geometry is exact (checked by tests/test_reprojection.py).

Panda3D lens convention (verified empirically): film offset y = cy - H/2, and the RAM image is
bottom-up (flip rows). Panda3D camera axes are x right, y forward, z up.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from panda3d.core import (
  AmbientLight,
  CardMaker,
  DirectionalLight,
  Fog,
  FrameBufferProperties,
  Geom,
  GeomNode,
  GeomTriangles,
  GeomVertexData,
  GeomVertexFormat,
  GeomVertexWriter,
  GraphicsOutput,
  GraphicsPipe,
  LMatrix4f,
  NodePath,
  PerspectiveLens,
  Texture,
  TextureStage,
  WindowProperties,
  loadPrcFileData,
)

from fsop.geometry.camera import CameraModel
from fsop.sim.randomization import SceneDR
from fsop.sim.render import venue
from fsop.sim.render.assets import cone_geom, metadrive_texture, procedural_asphalt
from fsop.track.rules import RULES
from fsop.track.schema import ConeColor, Track

_BASE = None

# Panda camera axes expressed in our camera-body frame (x fwd, y left, z up)
_M_BODY_FROM_PANDA = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _showbase():
  global _BASE
  if _BASE is None:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
      loadPrcFileData("", "load-display p3headlessgl\n")  # EGL: GPU driver or Mesa llvmpipe on servers
    loadPrcFileData("", "window-type offscreen\naudio-library-name null\nsync-video 0\n"
                        "notify-level-display error\nnotify-level-glgsg error\nframebuffer-multisample 0\n"
                        "textures-power-2 none\n")
    from direct.showbase.ShowBase import ShowBase

    _BASE = ShowBase()
    _BASE.disableMouse()
  return _BASE


def mat_from_T(T: np.ndarray) -> LMatrix4f:
  return LMatrix4f(*T.T.astype(np.float64).ravel())


class Panda3DRenderer:
  def __init__(self, cameras: dict[str, CameraModel], msaa: int = 0):
    self.base = _showbase()
    self.cameras = cameras
    self.buffers, self.textures, self.cam_nps = {}, {}, {}
    self.scene: NodePath | None = None
    self.sky: NodePath | None = None
    self.cone_nodes: list[NodePath] = []
    self._cone_models = {c: NodePath(cone_geom(c)) for c in ConeColor}
    self._cone_variants: dict[tuple, NodePath] = {}
    for name, cam in cameras.items():
      fb = FrameBufferProperties()
      fb.setRgbColor(True)
      fb.setRgbaBits(8, 8, 8, 0)
      fb.setDepthBits(24)
      if msaa:
        fb.setMultisamples(msaa)
      buf = self.base.graphicsEngine.makeOutput(self.base.pipe, f"buf_{name}", -2, fb,
                                                WindowProperties.size(cam.width, cam.height),
                                                GraphicsPipe.BFRefuseWindow, self.base.win.getGsg(), self.base.win)
      tex = Texture()
      buf.addRenderTexture(tex, GraphicsOutput.RTMCopyRam)
      np_cam = self.base.makeCamera(buf)
      lens = PerspectiveLens()
      lens.setFilmSize(cam.width, cam.height)
      lens.setFocalLength(cam.K[0, 0])
      lens.setFilmOffset(cam.K[0, 2] - cam.width / 2, cam.K[1, 2] - cam.height / 2)
      lens.setNearFar(0.1, 600.0)
      np_cam.node().setLens(lens)
      self.buffers[name], self.textures[name], self.cam_nps[name] = buf, tex, np_cam
    self.base.win.setActive(False)

  # ------------------------------------------------------------------------------------------
  def load_scene(self, track: Track, cones_xy: np.ndarray, cone_colors: list[ConeColor], dr: SceneDR) -> None:
    if self.scene is not None:
      self.scene.removeNode()
    rng = np.random.default_rng(dr.seed)
    root = self.base.render.attachNewNode("scene")
    self.scene = root
    for buf in self.buffers.values():
      buf.setClearColor((*dr.sky_color, 1.0))
    self.sky = venue.sky_dome(root, dr.sky_pano, yaw=dr.sky_yaw) if dr.sky_pano else None
    # ground
    pad = 150 if dr.venue else 60  # venue: the horizon is visible, so the ground must reach further
    lo = track.centerline.min(0) - pad
    hi = track.centerline.max(0) + pad
    if dr.ground_kind == "venue":
      tex = venue._np_texture(venue.venue_ground(rng, dr.ground_base, dr.ground_tint), "venue_ground")
      tile = venue.GROUND_TILE_M
    elif dr.ground_kind == "metadrive" and metadrive_texture("asphalt") is not None:
      tex = self.base.loader.loadTexture(str(metadrive_texture("asphalt")))
      tex.setWrapU(Texture.WM_repeat)
      tex.setWrapV(Texture.WM_repeat)
      tex.setMinfilter(Texture.FT_linear_mipmap_linear)
      tile = 8.0
    else:
      tex = procedural_asphalt(rng, base=dr.ground_base, contrast=dr.ground_contrast)
      tile = 6.0
    g = self._card(root, lo, hi, 0.0)
    g.setTexture(tex)
    g.setTexScale(TextureStage.getDefault(), (hi[0] - lo[0]) / tile, (hi[1] - lo[1]) / tile)
    if dr.ground_kind != "venue":
      g.setColorScale(*dr.ground_tint, 1.0)
    else:  # fine aggregate grain modulated on top of the large texture
      ts = TextureStage("grain")
      ts.setMode(TextureStage.MModulate)
      g.setTexture(ts, venue.grain_texture(rng))
      g.setTexScale(ts, (hi[0] - lo[0]) / 1.5, (hi[1] - lo[1]) / 1.5)
    if dr.grass_ring:
      gtex_path = metadrive_texture("grass")
      far_lo, far_hi = lo - 400, hi + 400
      grass = self._card(root, far_lo, far_hi, -0.02)
      if gtex_path is not None:
        gt = self.base.loader.loadTexture(str(gtex_path))
        gt.setWrapU(Texture.WM_repeat)
        gt.setWrapV(Texture.WM_repeat)
        grass.setTexture(gt)
        grass.setTexScale(TextureStage.getDefault(), (far_hi[0] - far_lo[0]) / 10, (far_hi[1] - far_lo[1]) / 10)
      else:
        grass.setColor(0.3, 0.45, 0.2, 1)
    if dr.boundary_lines:
      left, right = track.boundaries()
      for b in (left, right):
        self._strip(root, b, dr.line_width, dr.line_color, track.closed, rng, dr.line_gap_prob)
    # distractors: low boxes (barriers / tents / parked cars) away from the track
    from shapely.geometry import LineString, Point

    line = LineString(track.centerline)
    for _ in range(int(rng.integers(0, 25))):
      p = rng.uniform(lo, hi)
      if line.distance(Point(p)) < 8.0:
        continue
      size = rng.uniform([0.5, 0.5, 0.4], [6.0, 3.0, 2.5])
      self._box(root, p, size, rng.uniform(0, np.pi), tuple(rng.uniform(0.2, 0.9, 3)))
    people = venue.build_venue(root, track, rng) if dr.venue else np.zeros((0, 2))
    # cones
    self.cone_nodes = []
    for (x, y), c in zip(cones_xy, cone_colors, strict=True):
      n = root.attachNewNode("cone")
      self._cone_model(c, dr).instanceTo(n)
      n.setPos(float(x), float(y), 0.0)
      n.setH(float(rng.uniform(0, 360)))
      b = float(np.clip(1 + rng.normal(0, dr.cone_brightness_sigma), 0.6, 1.4))
      n.setColorScale(b, b, b, 1)
      self.cone_nodes.append(n)
    # lighting
    root.clearLight()
    sun = DirectionalLight("sun")
    sun.setColor((*(np.array(dr.sun_color) * dr.sun_intensity), 1))
    sun_np = root.attachNewNode(sun)
    az, el = dr.sun_azimuth, dr.sun_elevation
    sun_np.lookAt(-np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), -np.sin(el))
    amb = AmbientLight("amb")
    amb.setColor((dr.ambient, dr.ambient, dr.ambient * 1.05, 1))
    root.setLight(sun_np)
    root.setLight(root.attachNewNode(amb))
    if dr.shadow_alpha > 0:
      polys = venue.shadow_polys(np.asarray(cones_xy).reshape(-1, 2), RULES.small_cone.radius_bottom,
                                 RULES.small_cone.height, az, el)
      polys += venue.shadow_polys(people, 0.25, 1.7, az, el)
      venue.ground_shadows(root, polys, dr.shadow_alpha)
    if dr.fog_density > 0:
      fog = Fog("fog")
      fog.setColor(*dr.sky_color)
      fog.setExpDensity(dr.fog_density)
      root.setFog(fog)

  def _cone_model(self, c: ConeColor, dr: SceneDR) -> NodePath:
    if not dr.venue:
      return self._cone_models[c]
    key = (c, round(dr.cone_fade, 1), dr.cone_lettering)
    if key not in self._cone_variants:
      self._cone_variants[key] = NodePath(cone_geom(c, fade=key[1], lettering=key[2]))
    return self._cone_variants[key]

  def knock_cone(self, idx: int, direction: float) -> None:
    n = self.cone_nodes[idx]
    n.setHpr(np.rad2deg(direction), 90, 0)
    n.setZ(0.12)

  def render(self, T_world_cam: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if self.sky is not None:  # the panorama is at infinity: keep it centred on the camera
      t = next(iter(T_world_cam.values()))[:3, 3]
      self.sky.setPos(float(t[0]), float(t[1]), float(t[2]))
    for name, T in T_world_cam.items():
      Tp = T.copy()
      Tp[:3, :3] = T[:3, :3] @ _M_BODY_FROM_PANDA
      self.cam_nps[name].setMat(mat_from_T(Tp))
    self.base.graphicsEngine.renderFrame()
    out = {}
    for name in T_world_cam:
      tex = self.textures[name]
      img = np.frombuffer(tex.getRamImageAs("RGB").getData(), np.uint8)
      out[name] = img.reshape(tex.getYSize(), tex.getXSize(), 3)[::-1].copy()
    return out

  # ------------------------------------------------------------------------------------------
  @staticmethod
  def _card(root, lo, hi, z, cell: float = 8.0) -> NodePath:
    """Horizontal plane in the world XY plane with UVs 0..1 (upward normal), tessellated into ~``cell`` m
    quads: fixed-function fog is evaluated per vertex, so one huge quad would be fogged as a whole."""
    fmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData("ground", fmt, Geom.UHStatic)
    vw, nw = GeomVertexWriter(vdata, "vertex"), GeomVertexWriter(vdata, "normal")
    tw = GeomVertexWriter(vdata, "texcoord")
    nx = max(int(np.ceil((hi[0] - lo[0]) / cell)), 1)
    ny = max(int(np.ceil((hi[1] - lo[1]) / cell)), 1)
    for j in range(ny + 1):
      for i in range(nx + 1):
        u, v = i / nx, j / ny
        vw.addData3(float(lo[0] + u * (hi[0] - lo[0])), float(lo[1] + v * (hi[1] - lo[1])), float(z))
        nw.addData3(0, 0, 1)
        tw.addData2(u, v)
    tris = GeomTriangles(Geom.UHStatic)
    for j in range(ny):
      for i in range(nx):
        a = j * (nx + 1) + i
        tris.addVertices(a, a + 1, a + nx + 2)
        tris.addVertices(a, a + nx + 2, a + nx + 1)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("ground")
    node.addGeom(geom)
    return root.attachNewNode(node)

  @staticmethod
  def _strip(root, poly: np.ndarray, width: float, color, closed: bool, rng, gap_prob: float, z: float = 0.004):
    """Flat line (chalk/paint) along a polyline, with random gaps, as a lit horizontal triangle strip."""
    pts = np.vstack([poly, poly[:1]]) if closed else poly
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    a, b = pts + n * width / 2, pts - n * width / 2
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("line", fmt, Geom.UHStatic)
    vw, nw, cw = GeomVertexWriter(vdata, "vertex"), GeomVertexWriter(vdata, "normal"), GeomVertexWriter(vdata, "color")
    for p, q in zip(a, b, strict=True):
      for v in (p, q):
        vw.addData3(float(v[0]), float(v[1]), z)
        nw.addData3(0, 0, 1)
        cw.addData4(*color, 1.0)
    tris = GeomTriangles(Geom.UHStatic)
    seg = max(int(1.5 / max(np.linalg.norm(pts[1] - pts[0]), 1e-3)), 1)  # gap granularity ~1.5 m
    for i in range(len(pts) - 1):
      if gap_prob and (i // seg) % 7 == 0 and rng.random() < gap_prob:
        continue
      j = 2 * i
      tris.addVertices(j, j + 1, j + 2)
      tris.addVertices(j + 1, j + 3, j + 2)
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("line")
    node.addGeom(geom)
    np_ = root.attachNewNode(node)
    np_.setTwoSided(True)

  @staticmethod
  def _box(root, p, size, yaw, color) -> None:
    cm = CardMaker("face")
    box = root.attachNewNode("box")
    sx, sy, sz = size
    for h, (w, d) in ((0, (sx, sy)), (90, (sy, sx)), (180, (sx, sy)), (270, (sy, sx))):
      cm.setFrame(-w / 2, w / 2, 0, sz)
      f = box.attachNewNode(cm.generate())
      f.setH(h)
      f.setY(f, -d / 2)
    cm.setFrame(-sx / 2, sx / 2, -sy / 2, sy / 2)
    top = box.attachNewNode(cm.generate())
    top.setP(-90)
    top.setZ(sz)
    box.setPos(float(p[0]), float(p[1]), 0)
    box.setH(float(np.rad2deg(yaw)))
    box.setColor(*color, 1)
    box.setTwoSided(True)
