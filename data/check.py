from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from excavator_sim.common import get_paths


JOINT_ORDER = ["boom_joint", "arm_joint", "bucket_joint", "swing_joint"]


@dataclass
class CheckConfig:
    run_dirs: list[Path]
    joint_tol_rad: float
    min_duration_s: float
    max_duration_s: float
    min_stone_ratio: float
    min_sensor_hz: float


@dataclass
class RunMetrics:
    duration_s: float
    final_stones_in_truck: int
    camera_driver_hz: float
    camera_bucket_hz: float
    lidar_hz: float


def _load_table(run_dir: Path, stem: str) -> pd.DataFrame:
    parquet_path = run_dir / f"{stem}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    pickle_path = run_dir / f"{stem}.pkl"
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    raise FileNotFoundError(f"missing table for {stem}: {parquet_path} or {pickle_path}")


def _load_optional_table(run_dir: Path, stem: str, columns: list[str]) -> pd.DataFrame:
    try:
        return _load_table(run_dir, stem)
    except FileNotFoundError:
        return pd.DataFrame(columns=columns)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except Exception:
            pass
    return []


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


def _duration_from_meta_or_tables(meta: dict, proprio_df: pd.DataFrame, action_df: pd.DataFrame) -> float:
    record_window = meta.get("record_window", {})
    start_recv_ns = record_window.get("start_recv_ns")
    finish_recv_ns = record_window.get("finish_recv_ns")
    if start_recv_ns is not None and finish_recv_ns is not None:
        try:
            return float((int(finish_recv_ns) - int(start_recv_ns)) / 1e9)
        except Exception:
            pass

    values: list[int] = []
    if not proprio_df.empty and "proprio_recv_ns" in proprio_df.columns:
        values.extend([int(proprio_df["proprio_recv_ns"].min()), int(proprio_df["proprio_recv_ns"].max())])
    if not action_df.empty and "action_recv_ns" in action_df.columns:
        values.extend([int(action_df["action_recv_ns"].min()), int(action_df["action_recv_ns"].max())])
    if len(values) >= 2:
        return float((max(values) - min(values)) / 1e9)
    return 0.0


def _initial_joint_positions(proprio_df: pd.DataFrame) -> dict[str, float] | None:
    if proprio_df.empty:
        return None
    row = proprio_df.iloc[0]
    names = _as_list(row.get("proprio_name"))
    positions = _as_list(row.get("proprio_position"))
    if not names or not positions:
        return None
    out: dict[str, float] = {}
    for name, pos in zip(names, positions):
        if name in JOINT_ORDER:
            out[str(name)] = float(pos)
    return out if out else None


def _check_run(run_dir: Path, cfg: CheckConfig) -> tuple[list[str], RunMetrics]:
    issues: list[str] = []

    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return ["missing meta.json"], RunMetrics(0.0, 0, 0.0, 0.0, 0.0)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    proprio_df = _load_optional_table(run_dir, "proprio", ["stamp_ns", "proprio_recv_ns", "proprio_name", "proprio_position"])
    action_df = _load_optional_table(run_dir, "action", ["stamp_ns", "action_recv_ns"])
    stones_df = _load_optional_table(run_dir, "stones_in_truck", ["stamp_ns", "stones_recv_ns", "stones_count"])

    init_positions = _initial_joint_positions(proprio_df)
    if init_positions is None:
        issues.append("missing initial joint_states")
    else:
        missing_joints = [joint for joint in JOINT_ORDER if joint not in init_positions]
        if missing_joints:
            issues.append(f"missing init joints: {missing_joints}")
        bad_joints = [f"{joint}={init_positions[joint]:+.3f}" for joint in JOINT_ORDER if joint in init_positions and abs(init_positions[joint]) > cfg.joint_tol_rad]
        if bad_joints:
            issues.append(f"init joints out of range (+/-{cfg.joint_tol_rad:.2f} rad): {', '.join(bad_joints)}")

    final_stones_in_truck = 0
    if stones_df.empty:
        issues.append("missing stones_in_truck table")
    else:
        start_stones = int(stones_df.iloc[0].get("stones_count", -1))
        end_stones = int(stones_df.iloc[-1].get("stones_count", -1))
        final_stones_in_truck = end_stones
        if start_stones != 0:
            issues.append(f"start stones_in_truck is {start_stones}, expected 0")

        total_stones = meta.get("episode_meta", {}).get("stone_count")
        if total_stones is None:
            issues.append("missing episode_meta.stone_count")
        else:
            total_stones = int(total_stones)
            required = int(np.ceil(total_stones * cfg.min_stone_ratio))
            if end_stones < required:
                issues.append(f"end stones_in_truck is {end_stones}, expected >= {required} ({cfg.min_stone_ratio:.0%} of {total_stones})")

    duration_s = _duration_from_meta_or_tables(meta, proprio_df, action_df)
    if duration_s < cfg.min_duration_s:
        issues.append(f"duration too short: {duration_s:.2f}s < {cfg.min_duration_s:.2f}s")
    if duration_s > cfg.max_duration_s:
        issues.append(f"duration too long: {duration_s:.2f}s > {cfg.max_duration_s:.2f}s")

    hz_recv = meta.get("topic_hz_avg_recv", {})
    camera_driver_hz = float(hz_recv.get("camera_driver", 0.0) or 0.0)
    camera_bucket_hz = float(hz_recv.get("camera_bucket", 0.0) or 0.0)
    lidar_hz = float(hz_recv.get("lidar", 0.0) or 0.0)
    for key in ("camera_driver", "camera_bucket", "lidar"):
        hz = hz_recv.get(key)
        if hz is None:
            issues.append(f"missing topic_hz_avg_recv.{key}")
            continue
        hz = float(hz)
        if hz <= cfg.min_sensor_hz:
            issues.append(f"{key} hz too low: {hz:.2f} <= {cfg.min_sensor_hz:.2f}")

    return issues, RunMetrics(
        duration_s=duration_s,
        final_stones_in_truck=final_stones_in_truck,
        camera_driver_hz=camera_driver_hz,
        camera_bucket_hz=camera_bucket_hz,
        lidar_hz=lidar_hz,
    )


def parse_args() -> CheckConfig:
    parser = argparse.ArgumentParser(description="Check recorded runs against basic validity rules")
    parser.add_argument("run_dirs", nargs="*", help="Run directories or run_xxx names. Default: all runs under data/raw")
    parser.add_argument("--joint-tol-rad", type=float, default=0.1)
    parser.add_argument("--min-duration-s", type=float, default=20.0)
    parser.add_argument("--max-duration-s", type=float, default=50.0)
    parser.add_argument("--min-stone-ratio", type=float, default=0.08)
    parser.add_argument("--min-sensor-hz", type=float, default=15.0)
    args = parser.parse_args()
    return CheckConfig(
        run_dirs=_resolve_run_dirs(args.run_dirs),
        joint_tol_rad=float(args.joint_tol_rad),
        min_duration_s=float(args.min_duration_s),
        max_duration_s=float(args.max_duration_s),
        min_stone_ratio=float(args.min_stone_ratio),
        min_sensor_hz=float(args.min_sensor_hz),
    )


def main() -> None:
    cfg = parse_args()
    if not cfg.run_dirs:
        print("[check] no run directories found")
        return

    failed = []
    passed_metrics: list[tuple[str, RunMetrics]] = []
    checked = 0
    for run_dir in cfg.run_dirs:
        checked += 1
        try:
            issues, metrics = _check_run(run_dir, cfg)
        except Exception as exc:
            issues = [f"check failed: {exc}"]
            metrics = RunMetrics(0.0, 0, 0.0, 0.0, 0.0)
        if issues:
            failed.append((run_dir.name, issues))
        else:
            passed_metrics.append((run_dir.name, metrics))

    print(f"[check] checked runs: {checked}")
    print(f"[check] passed runs: {len(passed_metrics)}")
    print(f"[check] failed runs: {len(failed)}")
    if passed_metrics:
        camera_driver_vals = np.asarray([m.camera_driver_hz for _, m in passed_metrics], dtype=np.float64)
        camera_bucket_vals = np.asarray([m.camera_bucket_hz for _, m in passed_metrics], dtype=np.float64)
        lidar_vals = np.asarray([m.lidar_hz for _, m in passed_metrics], dtype=np.float64)
        duration_vals = np.asarray([m.duration_s for _, m in passed_metrics], dtype=np.float64)
        stones_vals = np.asarray([m.final_stones_in_truck for _, m in passed_metrics], dtype=np.float64)

        print("\n[check] passed-run summary")
        print(
            f"  camera_driver_hz: min={camera_driver_vals.min():.2f} max={camera_driver_vals.max():.2f} avg={camera_driver_vals.mean():.2f}"
        )
        print(
            f"  camera_bucket_hz: min={camera_bucket_vals.min():.2f} max={camera_bucket_vals.max():.2f} avg={camera_bucket_vals.mean():.2f}"
        )
        print(f"  lidar_hz:         min={lidar_vals.min():.2f} max={lidar_vals.max():.2f} avg={lidar_vals.mean():.2f}")
        print(
            f"  episode_length_s: min={duration_vals.min():.2f} max={duration_vals.max():.2f} avg={duration_vals.mean():.2f}"
        )
        print(
            f"  final_stones:     min={stones_vals.min():.0f} max={stones_vals.max():.0f} avg={stones_vals.mean():.2f}"
        )

    if not failed:
        print("[check] all runs passed")
        return

    for run_name, issues in failed:
        print(f"\n{run_name}")
        for issue in issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()
