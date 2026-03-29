from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load src/excavator_policy/config.yaml. Please install pyyaml.") from exc

    cfg_path = Path(path).expanduser().resolve() if path is not None else DEFAULT_CONFIG_PATH.resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping: {cfg_path}")
    cfg["_config_path"] = str(cfg_path)
    return cfg
