from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    raw_root: Path
    window: int = 8
    horizon: int = 1
    points_max: int = 4096


def _to_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            out = ast.literal_eval(v)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


class ExcavatorDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.runs = sorted([p for p in cfg.raw_root.glob("run_*") if p.is_dir()])
        self.index: List[Tuple[Path, int]] = []

        for run in self.runs:
            ts_file = run / "timestamps.parquet"
            if not ts_file.exists():
                continue
            ts_df = pd.read_parquet(ts_file)
            n = len(ts_df)
            for i in range(cfg.window - 1, max(cfg.window - 1, n - cfg.horizon)):
                self.index.append((run, i))

    def __len__(self):
        return len(self.index)

    def _load_points(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((self.cfg.points_max, 3), dtype=np.float32)
        pts = np.load(path)
        if pts.ndim != 2 or pts.shape[-1] < 3:
            return np.zeros((self.cfg.points_max, 3), dtype=np.float32)
        pts = pts[:, :3].astype(np.float32)
        if len(pts) >= self.cfg.points_max:
            sel = np.random.choice(len(pts), self.cfg.points_max, replace=False)
            return pts[sel]
        out = np.zeros((self.cfg.points_max, 3), dtype=np.float32)
        out[: len(pts)] = pts
        return out

    def _load_rgb(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((3, 64, 64), dtype=np.float32)
        img = np.load(path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = img[..., :3]
        else:
            return np.zeros((3, 64, 64), dtype=np.float32)
        img = img.astype(np.float32) / 255.0
        # Quick nearest-neighbor resize to 64x64 without extra deps.
        h, w = img.shape[:2]
        ys = (np.linspace(0, h - 1, 64)).astype(np.int32)
        xs = (np.linspace(0, w - 1, 64)).astype(np.int32)
        img = img[ys][:, xs]
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, idx: int):
        run_dir, end_idx = self.index[idx]
        ts_df = pd.read_parquet(run_dir / "timestamps.parquet")

        win_df = ts_df.iloc[end_idx - self.cfg.window + 1 : end_idx + 1]
        cur = win_df.iloc[-1]

        rgb_path = run_dir / str(cur.get("rgb_path", ""))
        lidar_path = run_dir / str(cur.get("lidar_path", ""))

        obs = {
            "rgb": torch.from_numpy(self._load_rgb(rgb_path)),
            "points": torch.from_numpy(self._load_points(lidar_path)),
            "proprio": torch.tensor(_to_list(cur.get("proprio_position", [])), dtype=torch.float32),
        }

        action_seq = ts_df.iloc[end_idx + 1 : end_idx + 1 + self.cfg.horizon]
        if len(action_seq) == 0:
            action = torch.zeros_like(obs["proprio"])
        else:
            action = torch.tensor(_to_list(action_seq.iloc[0].get("action_position", [])), dtype=torch.float32)

        return obs, action


def build_dataset(raw_root: str, window: int = 8, horizon: int = 1) -> ExcavatorDataset:
    return ExcavatorDataset(DatasetConfig(raw_root=Path(raw_root), window=window, horizon=horizon))
