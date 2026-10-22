import numpy as np
import pytest

from fsop.track.fixed_layouts import acceleration, skidpad
from fsop.track.generate import TrackGenCfg, generate_track
from fsop.track.io import from_cone_csv, load_json, save_json, to_cone_csv
from fsop.track.rules import RULES
from fsop.track.schema import ConeColor
from fsop.track.validate import validate


@pytest.mark.parametrize("seed", range(8))
def test_generated_tracks_pass_rules(seed):
  t = generate_track(seed)
  rep = validate(t)
  assert rep.passed, str(rep)
  assert RULES.trackdrive_lap_min.value <= t.length <= RULES.trackdrive_lap_max.value
  assert t.stats.max_straight <= 80.0
  assert t.stats.min_radius >= 4.5
  assert sum(c.color == ConeColor.BIG_ORANGE for c in t.cones) == 4


def test_autocross_config():
  t = generate_track(3, TrackGenCfg(discipline="autocross"))
  assert t.laps == 1 and validate(t).passed


def test_fixed_layouts():
  sp, ac = skidpad(), acceleration()
  assert validate(sp).passed and validate(ac).passed
  inner = [c for c in sp.cones if c.role.startswith("skidpad_inner")]
  assert len(inner) == 34
  r = np.hypot([c.x for c in inner], np.abs([c.y for c in inner]) - 9.125)
  np.testing.assert_allclose(r, 7.625, atol=1e-6)
  assert sp.route == ["entry", "right", "right", "left", "left", "exit"]


def test_json_and_csv_roundtrip(tmp_path):
  t = generate_track(11)
  save_json(t, tmp_path / "t.json")
  t2 = load_json(tmp_path / "t.json")
  np.testing.assert_allclose(t.centerline, t2.centerline, atol=1e-3)
  assert len(t.cones) == len(t2.cones)
  to_cone_csv(t, tmp_path / "t.csv")
  t3 = from_cone_csv(tmp_path / "t.csv")
  assert len(t3.cones) == len(t.cones)
  assert abs(t3.length - t.length) / t.length < 0.05
