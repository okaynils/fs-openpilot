import json

import numpy as np
import pytest
import zarr

from fsop.data.staging import stage_dataset
from fsop.sim.recorder import write_frames


def test_staging_materializes_external_frames_and_keeps_source_unchanged(tmp_path):
  frames = tmp_path / "original/frames.zarr"
  write_frames(frames, {"road": np.full((2, 6, 128, 256), 23, np.uint8)})
  source, destination = tmp_path / "labels", tmp_path / "scratch"
  episode = source / "ep"
  episode.mkdir(parents=True)
  meta = {"frames": 2, "frames_source": str(frames)}
  (episode / "meta.json").write_text(json.dumps(meta))
  (source / "index.json").write_text(json.dumps({"episodes": [{"id": "ep"}]}))
  for tag in ("wanted", "old"):
    cache = episode / f"cache_{tag}"
    cache.mkdir()
    for name in ("hidden", "final", "s16"):
      np.save(cache / f"{name}.npy", np.zeros(2))
  stage_dataset(source, destination, "wanted", 2, workers=1)
  assert json.loads((episode / "meta.json").read_text()) == meta
  assert "frames_source" not in json.loads((destination / "ep/meta.json").read_text())
  copied = zarr.open_group(str(destination / "ep/frames.zarr"), mode="r")
  assert np.all(copied["road"][:] == 23)
  assert (destination / "ep/cache_wanted/hidden.npy").exists()
  assert not (destination / "ep/cache_wanted/final.npy").exists()
  assert not (destination / "ep/cache_old").exists()
  assert (source / "ep/cache_wanted/final.npy").exists()


def test_staging_does_not_publish_index_if_frames_are_missing(tmp_path):
  source, destination = tmp_path / "labels", tmp_path / "scratch"
  (source / "ep").mkdir(parents=True)
  (source / "ep/meta.json").write_text(json.dumps({"frames_source": str(tmp_path / "missing")}))
  (source / "index.json").write_text(json.dumps({"episodes": [{"id": "ep"}]}))
  with pytest.raises(FileNotFoundError):
    stage_dataset(source, destination, "cache", 2, workers=1)
  assert not (destination / "index.json").exists()
  with pytest.raises(ValueError, match="separate"):
    stage_dataset(source, source, "cache", 2)
