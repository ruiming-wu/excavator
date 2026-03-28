from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default)).expanduser().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    assets: Path
    raw_data: Path
    aligned_data: Path
    logs: Path



def get_paths() -> ProjectPaths:
    root = _env_path("EXCAVATOR_ROOT", str(Path(__file__).resolve().parents[2]))
    paths = ProjectPaths(
        root=root,
        assets=_env_path("EXCAVATOR_ASSETS_DIR", str(root / "assets")),
        raw_data=_env_path("EXCAVATOR_DATA_RAW_DIR", str(root / "data" / "raw")),
        aligned_data=_env_path("EXCAVATOR_DATA_ALIGNED_DIR", str(root / "data" / "aligned")),
        logs=_env_path("EXCAVATOR_LOGS_DIR", str(root / "logs")),
    )
    for p in [paths.assets, paths.raw_data, paths.aligned_data, paths.logs]:
        p.mkdir(parents=True, exist_ok=True)
    return paths
