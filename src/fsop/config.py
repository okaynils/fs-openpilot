"""Config loading: dataclass schema <- YAML file <- dot-list CLI overrides (OmegaConf)."""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path

from omegaconf import OmegaConf


def load_config[T](schema: type[T], path: str | Path | None = None, overrides: list[str] | None = None) -> T:
  assert is_dataclass(schema)
  cfg = OmegaConf.structured(schema)
  if path is not None:
    file_cfg = OmegaConf.load(path)
    if "defaults" in file_cfg:  # simple inheritance: defaults: [other.yaml]
      for parent in file_cfg.pop("defaults"):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(path).parent / parent))
    cfg = OmegaConf.merge(cfg, file_cfg)
  if overrides:
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
  return OmegaConf.to_object(cfg)  # type: ignore[return-value]


def save_config(cfg, path: str | Path) -> None:
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  OmegaConf.save(OmegaConf.structured(cfg), path)
