from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    aligned_root: Path
    raw_root: Path
    run_glob: str = "run_*"
    allowed_runs: tuple[str, ...] | None = None
    image_height: int = 160
    image_width: int = 240
    point_count: int = 4096
    point_dim: int = 3
    joint_order: tuple[str, ...] = ("swing_joint", "boom_joint", "arm_joint", "bucket_joint")
    future_start: int = 1
    horizon: int = 16
    seed: int = 42


def _to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        try:
            out = ast.literal_eval(v)
            return list(out) if isinstance(out, (list, tuple)) else []
        except Exception:
            return []
    if hasattr(v, "tolist"):
        try:
            out = v.tolist()
            return list(out) if isinstance(out, (list, tuple)) else []
        except Exception:
            return []
    return []


def _named_vector(names, values, joint_order: Sequence[str]) -> np.ndarray:
    out = np.zeros(len(joint_order), dtype=np.float32)
    names_list = _to_list(names)
    values_list = _to_list(values)
    if len(names_list) == 0 or len(values_list) == 0:
        return out
    mapping = {str(n): float(v) for n, v in zip(names_list, values_list)}
    for i, joint in enumerate(joint_order):
        out[i] = mapping.get(joint, 0.0)
    return out


class ExcavatorDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        allowed_runs = set(cfg.allowed_runs) if cfg.allowed_runs else None
        self.runs = sorted(
            [
                p
                for p in cfg.aligned_root.glob(cfg.run_glob)
                if p.is_dir() and (allowed_runs is None or p.name in allowed_runs)
            ]
        )
        self.frames_by_run: Dict[Path, pd.DataFrame] = {}
        self.index: List[Tuple[Path, int]] = []

        min_len = cfg.future_start + cfg.horizon
        for run in self.runs:
            frames_path = run / "frames.parquet"
            if not frames_path.exists():
                continue
            frames_df = pd.read_parquet(frames_path).reset_index(drop=True)
            if len(frames_df) < min_len:
                continue
            self.frames_by_run[run] = frames_df
            max_start = len(frames_df) - cfg.future_start - cfg.horizon + 1
            for i in range(max_start):
                self.index.append((run, i))

        self.joint_dim = len(cfg.joint_order)
        self.horizon = cfg.horizon

    def __len__(self) -> int:
        return len(self.index)

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim != 3 or img.shape[-1] < 3:
            return np.zeros((3, self.cfg.image_height, self.cfg.image_width), dtype=np.float32)
        img = img[..., :3].astype(np.float32) / 255.0
        h, w = img.shape[:2]
        if h != self.cfg.image_height or w != self.cfg.image_width:
            ys = (np.linspace(0, h - 1, self.cfg.image_height)).astype(np.int32)
            xs = (np.linspace(0, w - 1, self.cfg.image_width)).astype(np.int32)
            img = img[ys][:, xs]
        return np.transpose(img, (2, 0, 1))

    def _load_rgb(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((3, self.cfg.image_height, self.cfg.image_width), dtype=np.float32)
        return self._resize_image(np.load(path))

    def _load_points(self, path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((self.cfg.point_count, self.cfg.point_dim), dtype=np.float32)
        pts = np.load(path)
        if pts.ndim != 2 or pts.shape[-1] < self.cfg.point_dim:
            return np.zeros((self.cfg.point_count, self.cfg.point_dim), dtype=np.float32)
        pts = pts[:, : self.cfg.point_dim].astype(np.float32)
        n = len(pts)
        if n >= self.cfg.point_count:
            sel = self.rng.choice(n, self.cfg.point_count, replace=False)
            return pts[sel]
        out = np.zeros((self.cfg.point_count, self.cfg.point_dim), dtype=np.float32)
        out[:n] = pts
        if n > 0:
            fill = self.rng.choice(n, self.cfg.point_count - n, replace=True)
            out[n:] = pts[fill]
        return out

    def __getitem__(self, idx: int):
        run_dir, cur_idx = self.index[idx]
        frames_df = self.frames_by_run[run_dir]
        cur = frames_df.iloc[cur_idx]
        tgt = frames_df.iloc[cur_idx + self.cfg.future_start : cur_idx + self.cfg.future_start + self.cfg.horizon]
        raw_run_dir = self.cfg.raw_root / run_dir.name

        driver_path = raw_run_dir / str(cur.get("camera_driver_path", ""))
        bucket_path = raw_run_dir / str(cur.get("camera_bucket_path", ""))
        lidar_path = raw_run_dir / str(cur.get("lidar_path", ""))

        current_state = _named_vector(cur.get("proprio_name", []), cur.get("proprio_position", []), self.cfg.joint_order)
        future_cmd = np.stack(
            [_named_vector(row.get("action_name", []), row.get("action_position", []), self.cfg.joint_order) for _, row in tgt.iterrows()],
            axis=0,
        ).astype(np.float32)

        obs = {
            "camera_driver": torch.from_numpy(self._load_rgb(driver_path)),
            "camera_bucket": torch.from_numpy(self._load_rgb(bucket_path)),
            "points": torch.from_numpy(self._load_points(lidar_path)),
            "current_state": torch.from_numpy(current_state),
        }
        action = torch.from_numpy(future_cmd)
        return obs, action


def build_dataset_from_config(data_cfg: dict, allowed_runs: Sequence[str] | None = None) -> ExcavatorDataset:
    cfg = DatasetConfig(
        aligned_root=Path(data_cfg["aligned_root"]),
        raw_root=Path(data_cfg["raw_root"]),
        run_glob=str(data_cfg.get("run_glob", "run_*")),
        allowed_runs=None if allowed_runs is None else tuple(str(x) for x in allowed_runs),
        image_height=int(data_cfg.get("image_height", 160)),
        image_width=int(data_cfg.get("image_width", 240)),
        point_count=int(data_cfg.get("point_count", 4096)),
        point_dim=int(data_cfg.get("point_dim", 3)),
        joint_order=tuple(data_cfg.get("joint_order", ["swing_joint", "boom_joint", "arm_joint", "bucket_joint"])),
        future_start=int(data_cfg.get("future_start", 1)),
        horizon=int(data_cfg.get("horizon", 16)),
        seed=int(data_cfg.get("seed", 42)),
    )
    return ExcavatorDataset(cfg)
