"""Download openpilot's driving model at a pinned commit (MIT licensed, commaai/openpilot)."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

COMMIT = "b361e952c9da023412c6744278cc9421b2c90b8c"
SHA256 = "659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b"
URL = (f"https://media.githubusercontent.com/media/commaai/openpilot/{COMMIT}/openpilot/selfdrive/modeld/models/"
       "driving_supercombo.onnx")
DEFAULT = Path("artifacts/models/driving_supercombo_b361e952.onnx")


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def fetch(out: str | Path = DEFAULT) -> Path:
  out = Path(out)
  if not out.exists() or sha256(out) != SHA256:
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, out)
  digest = sha256(out)
  if digest != SHA256:
    raise RuntimeError(f"sha256 mismatch for {out}: {digest}")
  (out.parent / "manifest.json").write_text(json.dumps({"source": URL, "commit": COMMIT, "sha256": digest,
                                                         "license": "MIT (commaai/openpilot)"}, indent=1))
  return out
