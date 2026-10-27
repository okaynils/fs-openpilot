"""Config loading: dataclass schema <- YAML file <- dot-list CLI overrides (OmegaConf)."""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path

from omegaconf import OmegaConf


def load_config[T](schema: type[T], path: str | Path | None = None, overrides: list[str] | None = None) -> T:
  assert is_dataclass(schema)
  cfg = OmegaConf.structured(schema)
  if path is not None:
    cfg = OmegaConf.merge(cfg, _load_with_defaults(Path(path)))
  if overrides:
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
  return OmegaConf.to_object(cfg)  # type: ignore[return-value]


def _load_with_defaults(path: Path):
  """YAML with simple (recursive) inheritance: ``defaults: [other.yaml]`` relative to the file."""
  file_cfg = OmegaConf.load(path)
  if "defaults" not in file_cfg:
    return file_cfg
  parents = [_load_with_defaults(path.parent / p) for p in file_cfg.pop("defaults")]
  return OmegaConf.merge(*parents, file_cfg)


def save_config(cfg, path: str | Path) -> None:
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  OmegaConf.save(OmegaConf.structured(cfg), path)
