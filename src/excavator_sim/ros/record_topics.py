from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2

from excavator_sim.common import get_paths


@dataclass
class TopicRow:
    stamp_ns: int
    recv_ns: int
    payload: Dict[str, object] = field(default_factory=dict)


class Recorder(Node):
    def __init__(self, run_dir: Path):
        super().__init__("excavator_recorder")
        self.run_dir = run_dir
        self.rgb_dir = run_dir / "rgb"
        self.lidar_dir = run_dir / "lidar"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.lidar_dir.mkdir(parents=True, exist_ok=True)

        self.rgb_rows: List[TopicRow] = []
        self.lidar_rows: List[TopicRow] = []
        self.proprio_rows: List[TopicRow] = []
        self.action_rows: List[TopicRow] = []

        self._counter = 0
        self._last_log_ns = 0

        self.create_subscription(Image, "/excavator/camera_front/rgb", self.on_rgb, 10)
        self.create_subscription(PointCloud2, "/excavator/lidar/points", self.on_lidar, 10)
        self.create_subscription(JointState, "/excavator/joint_states", self.on_joint, 50)
        self.create_subscription(JointState, "/excavator/cmd_joint", self.on_cmd, 50)

    @staticmethod
    def _stamp_ns(msg) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _recv_ns(self) -> int:
        now = self.get_clock().now().nanoseconds
        return int(now)

    def _write_rgb(self, msg: Image, index: int) -> str:
        h, w = int(msg.height), int(msg.width)
        enc = msg.encoding.lower()
        np_data = np.frombuffer(msg.data, dtype=np.uint8)

        if enc in {"rgb8", "bgr8"}:
            img = np_data.reshape(h, w, 3)
        elif enc == "rgba8":
            img = np_data.reshape(h, w, 4)
        elif enc in {"mono8", "8uc1"}:
            img = np_data.reshape(h, w)
        else:
            img = np_data.copy()

        rel = f"rgb/{index:06d}.npy"
        np.save(self.run_dir / rel, img)
        return rel

    def _write_lidar(self, msg: PointCloud2, index: int) -> str:
        pts = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float32)
        rel = f"lidar/{index:06d}.npy"
        np.save(self.run_dir / rel, pts)
        return rel

    def on_rgb(self, msg: Image):
        idx = len(self.rgb_rows)
        rel_path = self._write_rgb(msg, idx)
        row = TopicRow(self._stamp_ns(msg), self._recv_ns(), {"path": rel_path, "encoding": msg.encoding})
        self.rgb_rows.append(row)
        self._tick()

    def on_lidar(self, msg: PointCloud2):
        idx = len(self.lidar_rows)
        rel_path = self._write_lidar(msg, idx)
        row = TopicRow(self._stamp_ns(msg), self._recv_ns(), {"path": rel_path, "frame_id": msg.header.frame_id})
        self.lidar_rows.append(row)
        self._tick()

    def on_joint(self, msg: JointState):
        row = TopicRow(
            self._stamp_ns(msg),
            self._recv_ns(),
            {
                "name": list(msg.name),
                "position": list(msg.position),
                "velocity": list(msg.velocity),
                "effort": list(msg.effort),
            },
        )
        self.proprio_rows.append(row)
        self._tick()

    def on_cmd(self, msg: JointState):
        row = TopicRow(
            self._stamp_ns(msg),
            self._recv_ns(),
            {
                "name": list(msg.name),
                "position": list(msg.position),
                "velocity": list(msg.velocity),
                "effort": list(msg.effort),
            },
        )
        self.action_rows.append(row)
        self._tick()

    def _tick(self):
        self._counter += 1
        now_ns = self._recv_ns()
        if now_ns - self._last_log_ns > 1_000_000_000:
            self.get_logger().info(
                f"rgb={len(self.rgb_rows)} lidar={len(self.lidar_rows)} proprio={len(self.proprio_rows)} action={len(self.action_rows)}"
            )
            self._last_log_ns = now_ns

    @staticmethod
    def _rows_to_df(rows: List[TopicRow], prefix: str) -> pd.DataFrame:
        data = []
        for r in rows:
            item = {"stamp_ns": r.stamp_ns, "recv_ns": r.recv_ns}
            for k, v in r.payload.items():
                item[f"{prefix}_{k}"] = v
            data.append(item)
        if not data:
            return pd.DataFrame(columns=["stamp_ns", "recv_ns"])
        return pd.DataFrame(data).sort_values("stamp_ns").reset_index(drop=True)

    @staticmethod
    def _nearest_asof(base: pd.DataFrame, target: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if base.empty:
            return base
        if target.empty:
            return base.copy()
        return pd.merge_asof(
            base.sort_values("stamp_ns"),
            target.sort_values("stamp_ns"),
            on="stamp_ns",
            direction="nearest",
            suffixes=("", suffix),
        )

    def flush(self):
        rgb_df = self._rows_to_df(self.rgb_rows, "rgb")
        lidar_df = self._rows_to_df(self.lidar_rows, "lidar")
        proprio_df = self._rows_to_df(self.proprio_rows, "proprio")
        action_df = self._rows_to_df(self.action_rows, "action")

        # Unified timeline anchored by action timestamps.
        aligned = action_df[["stamp_ns"]].copy() if not action_df.empty else proprio_df[["stamp_ns"]].copy()
        aligned = self._nearest_asof(aligned, rgb_df, "_rgb")
        aligned = self._nearest_asof(aligned, lidar_df, "_lidar")
        aligned = self._nearest_asof(aligned, proprio_df, "_proprio")
        aligned = self._nearest_asof(aligned, action_df, "_action")

        meta = {
            "topics": {
                "rgb": "/excavator/camera_front/rgb",
                "lidar": "/excavator/lidar/points",
                "joint_states": "/excavator/joint_states",
                "cmd_joint": "/excavator/cmd_joint",
            },
            "counts": {
                "rgb": len(rgb_df),
                "lidar": len(lidar_df),
                "proprio": len(proprio_df),
                "action": len(action_df),
                "aligned": len(aligned),
            },
        }

        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        proprio_df.to_parquet(self.run_dir / "proprio.parquet", index=False)
        action_df.to_parquet(self.run_dir / "action.parquet", index=False)
        aligned.to_parquet(self.run_dir / "timestamps.parquet", index=False)
        self.get_logger().info(f"saved run: {self.run_dir}")


def _next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not existing:
        return base_dir / "run_000"
    last = existing[-1].name.split("_")[-1]
    try:
        idx = int(last) + 1
    except ValueError:
        idx = len(existing)
    return base_dir / f"run_{idx:03d}"


def parse_args():
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Record excavator ROS2 topics")
    parser.add_argument("--out-dir", default="", help="Run directory (default: auto create under data/raw)")
    parser.add_argument("--base-dir", default=str(paths.raw_data), help="Base directory for run_xxx")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    run_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _next_run_dir(Path(args.base_dir))
    node = Recorder(run_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.flush()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
