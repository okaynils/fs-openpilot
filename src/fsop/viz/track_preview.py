"""Static (matplotlib) track previews for docs / quick inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fsop.track.schema import CONE_VIZ_RGB, ConeColor, Track  # noqa: E402
from fsop.track.validate import validate  # noqa: E402


def draw_track(ax, track: Track, show_center: bool = True, title: bool = True) -> None:
  left, right = track.boundaries()
  if track.discipline in ("trackdrive", "autocross"):
    from shapely.geometry import Polygon

    outer, inner = (left, right) if Polygon(left).area > Polygon(right).area else (right, left)
    ax.fill(outer[:, 0], outer[:, 1], color="#d9d9d9", zorder=0)
    ax.fill(inner[:, 0], inner[:, 1], color="white", zorder=0)
  if show_center:
    ax.plot(track.centerline[:, 0], track.centerline[:, 1], "--", lw=0.6, color="#888888", zorder=1)
  for color in ConeColor:
    pts = np.array([[c.x, c.y] for c in track.cones if c.color == color])
    if len(pts):
      size = 14 if color == ConeColor.BIG_ORANGE else 6
      ax.scatter(pts[:, 0], pts[:, 1], s=size, c=[np.array(CONE_VIZ_RGB[color]) / 255], zorder=3,
                 edgecolors="black" if color == ConeColor.BIG_ORANGE else "none", linewidths=0.4)
  sx, sy, syaw = track.start_pose
  ax.arrow(sx, sy, 3 * np.cos(syaw), 3 * np.sin(syaw), width=0.4, color="#d62728", zorder=4)
  ax.set_aspect("equal")
  ax.set_xticks([])
  ax.set_yticks([])
  if title:
    rep = validate(track)
    st = track.stats
    ax.set_title(f"{track.id}  {st.length:.0f} m  Rmin {st.min_radius:.1f} m\n"
                 f"straight {st.max_straight:.0f} m  {st.n_cones} cones  {'PASS' if rep.passed else 'FAIL'}",
                 fontsize=7, color="black" if rep.passed else "red")


def gallery(tracks: list[Track], out: str | Path, cols: int = 4) -> Path:
  rows = int(np.ceil(len(tracks) / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.4))
  axes = np.atleast_1d(axes).ravel()
  for ax, t in zip(axes, tracks, strict=False):
    draw_track(ax, t)
  for ax in axes[len(tracks):]:
    ax.axis("off")
  fig.tight_layout()
  out = Path(out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=130)
  plt.close(fig)
  return out
