from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from excavator_sim.common import get_paths


MODALITY_SPECS: dict[str, dict[str, object]] = {
    "camera_driver": {
        "recv_col": "driver_recv_ns",
        "stamp_col": "stamp_ns",
        "path_col": "driver_path",
        "columns": ["stamp_ns", "driver_recv_ns", "driver_path", "driver_encoding"],
    },
    "camera_bucket": {
        "recv_col": "bucket_recv_ns",
        "stamp_col": "stamp_ns",
        "path_col": "bucket_path",
        "columns": ["stamp_ns", "bucket_recv_ns", "bucket_path", "bucket_encoding"],
    },
    "lidar": {
        "recv_col": "lidar_recv_ns",
        "stamp_col": "stamp_ns",
        "path_col": "lidar_path",
        "columns": ["stamp_ns", "lidar_recv_ns", "lidar_path", "lidar_frame_id"],
    },
    "proprio": {
        "recv_col": "proprio_recv_ns",
        "stamp_col": "stamp_ns",
        "columns": [
            "stamp_ns",
            "proprio_recv_ns",
            "proprio_name",
            "proprio_position",
            "proprio_velocity",
            "proprio_effort",
        ],
    },
    "action": {
        "recv_col": "action_recv_ns",
        "stamp_col": "stamp_ns",
        "columns": [
            "stamp_ns",
            "action_recv_ns",
            "action_name",
            "action_position",
            "action_velocity",
            "action_effort",
        ],
    },
    "stones_in_truck": {
        "recv_col": "stones_recv_ns",
        "stamp_col": "stamp_ns",
        "columns": ["stamp_ns", "stones_recv_ns", "stones_count"],
    },
}

STRICT_WINDOW_MODALITIES = {"camera_driver", "camera_bucket", "lidar", "proprio", "action"}


@dataclass
class AlignConfig:
    run_dirs: list[Path]
    axis: str
    tolerance_ms: float
    out_root: Path


@dataclass
class MatchStats:
    max_abs_delta_ms: float = 0.0


def _load_table(run_dir: Path, stem: str) -> pd.DataFrame:
    parquet_path = run_dir / f"{stem}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    pickle_path = run_dir / f"{stem}.pkl"
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    raise FileNotFoundError(f"missing table for {stem}: {parquet_path} or {pickle_path}")


def _resolve_run_dirs(run_args: list[str]) -> list[Path]:
    paths = get_paths()
    if not run_args:
        return sorted(p for p in paths.raw_data.glob("run_*") if p.is_dir())

    resolved: list[Path] = []
    for arg in run_args:
        p = Path(arg).expanduser()
        if p.exists():
            resolved.append(p.resolve())
            continue
        candidate = paths.raw_data / arg
        if candidate.exists():
            resolved.append(candidate.resolve())
            continue
        raise FileNotFoundError(f"run dir not found: {arg}")
    return resolved


def _canonical_name(stem: str) -> str:
    if stem in MODALITY_SPECS:
        return stem
    aliases = {
        "driver": "camera_driver",
        "bucket": "camera_bucket",
        "joint_states": "proprio",
        "cmd_joint": "action",
        "stones": "stones_in_truck",
    }
    if stem in aliases:
        return aliases[stem]
    raise KeyError(f"unsupported modality: {stem}")


def _prepare_table(run_dir: Path, modality: str) -> pd.DataFrame:
    spec = MODALITY_SPECS[modality]
    stem = modality
    if modality == "camera_driver":
        stem = "camera_driver"
    elif modality == "camera_bucket":
        stem = "camera_bucket"
    df = _load_table(run_dir, stem)
    recv_col = str(spec["recv_col"])
    if recv_col not in df.columns:
        raise ValueError(f"{run_dir.name}:{modality} missing recv column {recv_col}")
    use_cols = [c for c in spec["columns"] if c in df.columns]
    df = df[use_cols].copy()
    df = df.sort_values(recv_col).drop_duplicates(subset=[recv_col], keep="first").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{run_dir.name}:{modality} table is empty")
    return df


def _nearest_index(sorted_values: np.ndarray, target: int) -> Tuple[int, int]:
    pos = int(np.searchsorted(sorted_values, target))
    candidates: list[int] = []
    if pos < len(sorted_values):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return -1, 10**18
    best_idx = min(candidates, key=lambda i: abs(int(sorted_values[i]) - target))
    best_delta = int(sorted_values[best_idx]) - int(target)
    return best_idx, best_delta


def _frame_key(modality: str, col: str) -> str:
    if col.startswith(f"{modality}_"):
        return col
    if modality == "camera_driver" and col.startswith("driver_"):
        return f"camera_{col}"
    if modality == "camera_bucket" and col.startswith("bucket_"):
        return f"camera_{col}"
    return f"{modality}_{col}"


def _match_all_modalities(
    axis_df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    axis_name: str,
    tolerance_ns: int,
) -> tuple[pd.DataFrame, Dict[str, MatchStats]]:
    axis_spec = MODALITY_SPECS[axis_name]
    axis_recv_col = str(axis_spec["recv_col"])
    axis_stamp_col = str(axis_spec["stamp_col"])
    axis_path_col = axis_spec.get("path_col")
    axis_recv = axis_df[axis_recv_col].to_numpy(dtype=np.int64)

    helper_arrays: dict[str, np.ndarray] = {}
    stats: dict[str, MatchStats] = {}
    for modality, df in tables.items():
        recv_col = str(MODALITY_SPECS[modality]["recv_col"])
        helper_arrays[modality] = df[recv_col].to_numpy(dtype=np.int64)
        stats[modality] = MatchStats()

    rows: list[dict[str, object]] = []
    for axis_idx, axis_row in axis_df.iterrows():
        axis_recv_ns = int(axis_row[axis_recv_col])
        out: dict[str, object] = {
            "frame_idx": int(axis_idx),
            "axis_modality": axis_name,
            "axis_recv_ns": axis_recv_ns,
            "axis_stamp_ns": int(axis_row.get(axis_stamp_col, 0) or 0),
        }
        if axis_path_col is not None and axis_path_col in axis_row:
            out[f"{axis_name}_path"] = axis_row[axis_path_col]
        out[f"delta_ms_{axis_name}"] = 0.0

        for col in axis_df.columns:
            if col in {axis_recv_col, axis_stamp_col, axis_path_col}:
                continue
            out[_frame_key(axis_name, col)] = axis_row[col]

        for modality, df in tables.items():
            if modality == axis_name:
                recv_col = str(MODALITY_SPECS[modality]["recv_col"])
                stamp_col = str(MODALITY_SPECS[modality]["stamp_col"])
                out[f"{modality}_recv_ns"] = int(axis_row[recv_col])
                out[f"{modality}_stamp_ns"] = int(axis_row.get(stamp_col, 0) or 0)
                continue

            matched_idx, delta_ns = _nearest_index(helper_arrays[modality], axis_recv_ns)
            if matched_idx < 0:
                raise ValueError(f"axis frame {axis_idx} cannot find {modality}")
            abs_delta_ns = abs(delta_ns)
            if modality in STRICT_WINDOW_MODALITIES and abs_delta_ns > tolerance_ns:
                raise ValueError(
                    f"axis frame {axis_idx} cannot match {modality} within ±{tolerance_ns / 1e6:.1f}ms "
                    f"(got {abs_delta_ns / 1e6:.1f}ms)"
                )

            stats[modality].max_abs_delta_ms = max(stats[modality].max_abs_delta_ms, abs_delta_ns / 1e6)
            row = df.iloc[matched_idx]
            recv_col = str(MODALITY_SPECS[modality]["recv_col"])
            stamp_col = str(MODALITY_SPECS[modality]["stamp_col"])
            out[f"{modality}_recv_ns"] = int(row[recv_col])
            out[f"{modality}_stamp_ns"] = int(row.get(stamp_col, 0) or 0)
            out[f"delta_ms_{modality}"] = float(delta_ns / 1e6)
            for col in df.columns:
                if col in {recv_col, stamp_col}:
                    continue
                out[_frame_key(modality, col)] = row[col]

        rows.append(out)

    frames_df = pd.DataFrame(rows)
    if not frames_df.empty:
        total_span_ns = int(axis_recv[-1] - axis_recv[0]) if len(axis_recv) > 1 else 0
        aligned_hz = 0.0 if total_span_ns <= 0 else float((len(frames_df) - 1) / (total_span_ns / 1e9))
        frames_df.attrs["aligned_hz"] = aligned_hz
    else:
        frames_df.attrs["aligned_hz"] = 0.0
    return frames_df, stats


def _summarize_valid(run_name: str, frames_df: pd.DataFrame, stats: Dict[str, MatchStats], axis_name: str) -> str:
    duration_s = 0.0
    if len(frames_df) >= 2:
        duration_s = float((int(frames_df["axis_recv_ns"].iloc[-1]) - int(frames_df["axis_recv_ns"].iloc[0])) / 1e9)
    aligned_hz = float(frames_df.attrs.get("aligned_hz", 0.0))
    summary_parts = [
        f"frames={len(frames_df)}",
        f"duration={duration_s:.2f}s",
        f"freq={aligned_hz:.2f}Hz",
        f"axis={axis_name}",
    ]
    for modality in ["camera_driver", "camera_bucket", "proprio", "action", "stones_in_truck"]:
        if modality in stats:
            summary_parts.append(f"max|dt|_{modality}={stats[modality].max_abs_delta_ms:.1f}ms")
    if "stones_in_truck" in stats:
        summary_parts.append("stones_in_truck=nearest-no-limit")
    if axis_name != "lidar" and "lidar" in stats:
        summary_parts.append(f"max|dt|_lidar={stats['lidar'].max_abs_delta_ms:.1f}ms")
    return f"[align] VALID {run_name}: " + ", ".join(summary_parts)


def _write_frames(out_dir: Path, frames_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.parquet"
    frames_df.to_parquet(frames_path, index=False)


def _process_run(run_dir: Path, cfg: AlignConfig) -> tuple[bool, str]:
    axis_name = _canonical_name(cfg.axis)
    tolerance_ns = int(cfg.tolerance_ms * 1e6)

    try:
        tables = {modality: _prepare_table(run_dir, modality) for modality in MODALITY_SPECS.keys()}
        axis_df = tables[axis_name]
        frames_df, stats = _match_all_modalities(axis_df=axis_df, tables=tables, axis_name=axis_name, tolerance_ns=tolerance_ns)
        out_dir = cfg.out_root / run_dir.name
        _write_frames(out_dir, frames_df)

        summary = {
            "run_id": run_dir.name,
            "axis_modality": axis_name,
            "tolerance_ms": cfg.tolerance_ms,
            "frames": int(len(frames_df)),
            "duration_s": 0.0 if len(frames_df) < 2 else float((int(frames_df["axis_recv_ns"].iloc[-1]) - int(frames_df["axis_recv_ns"].iloc[0])) / 1e9),
            "aligned_hz": float(frames_df.attrs.get("aligned_hz", 0.0)),
            "max_abs_delta_ms": {k: float(v.max_abs_delta_ms) for k, v in stats.items()},
        }
        (out_dir / "align_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return True, _summarize_valid(run_dir.name, frames_df, stats, axis_name)
    except Exception as exc:
        out_dir = cfg.out_root / run_dir.name
        frames_path = out_dir / "frames.parquet"
        if frames_path.exists():
            frames_path.unlink()
        return False, f"[align] INVALID {run_dir.name}: {exc}"


def parse_args() -> AlignConfig:
    parser = argparse.ArgumentParser(description="Align raw runs by nearest recv_ns and export frames.parquet")
    parser.add_argument("run_dirs", nargs="*", help="Run directories or run_xxx names. Default: all runs under data/raw")
    parser.add_argument("--axis", default="lidar", choices=sorted(MODALITY_SPECS.keys()))
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument("--out-dir", default="", help="Aligned output root. Default: data/aligned")
    args = parser.parse_args()

    paths = get_paths()
    out_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else paths.aligned_data
    return AlignConfig(
        run_dirs=_resolve_run_dirs(args.run_dirs),
        axis=str(args.axis),
        tolerance_ms=float(args.tolerance_ms),
        out_root=out_root,
    )


def main() -> None:
    cfg = parse_args()
    if not cfg.run_dirs:
        print("[align] no run directories found")
        return

    valid = 0
    invalid = 0
    for run_dir in cfg.run_dirs:
        ok, message = _process_run(run_dir, cfg)
        print(message)
        if ok:
            valid += 1
        else:
            invalid += 1

    print(f"[align] checked runs: {len(cfg.run_dirs)}")
    print(f"[align] valid runs: {valid}")
    print(f"[align] invalid runs: {invalid}")


if __name__ == "__main__":
    main()
