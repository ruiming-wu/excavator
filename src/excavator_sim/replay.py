from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from excavator_sim.common import get_paths


class ReplayNode(Node):
    def __init__(self, action_df: pd.DataFrame, speed: float):
        super().__init__("excavator_replay")
        self.action_df = action_df.reset_index(drop=True)
        self.speed = speed
        self.pub = self.create_publisher(JointState, "/excavator/cmd_joint", 10)

    def run(self):
        if self.action_df.empty:
            self.get_logger().warning("action parquet is empty")
            return

        t0 = int(self.action_df.iloc[0]["stamp_ns"])
        for i, row in self.action_df.iterrows():
            if i > 0:
                prev = int(self.action_df.iloc[i - 1]["stamp_ns"])
                curr = int(row["stamp_ns"])
                dt = max(0.0, (curr - prev) / 1e9 / max(self.speed, 1e-6))
                self.create_rate(max(1.0 / max(dt, 1e-3), 1.0)).sleep()

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(row.get("action_name", [])) if isinstance(row.get("action_name", []), list) else []

            pos = row.get("action_position", [])
            vel = row.get("action_velocity", [])
            eff = row.get("action_effort", [])
            msg.position = list(pos) if isinstance(pos, list) else []
            msg.velocity = list(vel) if isinstance(vel, list) else []
            msg.effort = list(eff) if isinstance(eff, list) else []

            self.pub.publish(msg)
            elapsed = (int(row["stamp_ns"]) - t0) / 1e9
            self.get_logger().info(f"replay step={i+1}/{len(self.action_df)} t={elapsed:.3f}s")
            rclpy.spin_once(self, timeout_sec=0.0)


def parse_args():
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Replay recorded cmd_joint trajectory")
    parser.add_argument("--run-dir", required=True, help="Path to run_xxx directory")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--action-file", default="action.parquet")
    parser.add_argument("--raw-base", default=str(paths.raw_data))
    return parser.parse_args()


def _resolve_run_dir(run_dir: str, raw_base: str) -> Path:
    path = Path(run_dir)
    if path.exists():
        return path.resolve()
    alt = Path(raw_base) / run_dir
    return alt.resolve()


def main():
    args = parse_args()
    run_dir = _resolve_run_dir(args.run_dir, args.raw_base)
    action_path = run_dir / args.action_file
    if not action_path.exists():
        raise FileNotFoundError(f"missing action file: {action_path}")

    action_df = pd.read_parquet(action_path)
    rclpy.init()
    node = ReplayNode(action_df=action_df, speed=args.speed)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
