from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Int32, String

from excavator_sim.common import get_paths


def _pointcloud_xyz_array(msg: PointCloud2) -> np.ndarray:
    pts = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    first = pts[0]
    if hasattr(first, "dtype") and getattr(first.dtype, "names", None):
        x = np.asarray([float(p["x"]) for p in pts], dtype=np.float32)
        y = np.asarray([float(p["y"]) for p in pts], dtype=np.float32)
        z = np.asarray([float(p["z"]) for p in pts], dtype=np.float32)
        return np.stack([x, y, z], axis=1)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


@dataclass
class TopicRow:
    stamp_ns: int
    recv_ns: int
    payload: Dict[str, object] = field(default_factory=dict)


class Recorder(Node):
    def __init__(self, base_dir: Path, out_dir: Optional[Path] = None):
        super().__init__("excavator_recorder")
        self.base_dir = base_dir
        self.out_dir = out_dir
        self.active_run_dir: Optional[Path] = None
        self.run_index = 0
        self.recording = False

        self.latest_ready = False
        self.latest_episode_meta: Dict[str, object] = {}

        self.driver_rows: List[TopicRow] = []
        self.bucket_rows: List[TopicRow] = []
        self.lidar_rows: List[TopicRow] = []
        self.proprio_rows: List[TopicRow] = []
        self.action_rows: List[TopicRow] = []
        self.stones_rows: List[TopicRow] = []
        self.record_control_rows: List[TopicRow] = []
        self.record_start_recv_ns: Optional[int] = None
        self.record_finish_recv_ns: Optional[int] = None

        self._counter = 0
        self._last_log_ns = 0
        self.table_format = "parquet"
        self.status_publisher = self.create_publisher(String, "/excavator/record_status", 10)

        latched_qos = QoSProfile(depth=1)
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched_qos.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(Image, "/excavator/camera_driver/rgb", self.on_driver_rgb, 10)
        self.create_subscription(Image, "/excavator/camera_bucket/rgb", self.on_bucket_rgb, 10)
        self.create_subscription(PointCloud2, "/excavator/lidar/points", self.on_lidar, 10)
        self.create_subscription(JointState, "/excavator/joint_states", self.on_joint, 50)
        self.create_subscription(JointState, "/excavator/cmd_joint", self.on_cmd, 50)
        self.create_subscription(Int32, "/excavator/stones_in_truck", self.on_stones_in_truck, 10)
        self.create_subscription(Int32, "/excavator/record_control", self.on_record_control, 10)
        self.create_subscription(Bool, "/excavator/ready", self.on_ready, latched_qos)
        self.create_subscription(String, "/excavator/episode_meta", self.on_episode_meta, latched_qos)

        self.get_logger().info("waiting for /excavator/record_control (1=start, 2=finish)")
        self._publish_status()

    @staticmethod
    def _stamp_ns(msg) -> int:
        if hasattr(msg, "header"):
            return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        return 0

    def _recv_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _reset_buffers(self) -> None:
        self.driver_rows = []
        self.bucket_rows = []
        self.lidar_rows = []
        self.proprio_rows = []
        self.action_rows = []
        self.stones_rows = []
        self.record_control_rows = []
        self.record_start_recv_ns = None
        self.record_finish_recv_ns = None
        self._counter = 0
        self._last_log_ns = 0

    def _prepare_run_dir(self) -> Path:
        if self.out_dir is not None and self.run_index == 0:
            run_dir = self.out_dir
        else:
            run_dir = _next_run_dir(self.base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("camera_driver", "camera_bucket", "lidar"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.run_index += 1
        return run_dir

    def _begin_run(self) -> None:
        if self.recording:
            self.get_logger().warning("record start ignored: already recording")
            return
        self._reset_buffers()
        self.active_run_dir = self._prepare_run_dir()
        self.recording = True
        self.record_start_recv_ns = self._recv_ns()
        self.record_control_rows.append(
            TopicRow(stamp_ns=0, recv_ns=self.record_start_recv_ns, payload={"command": 1, "label": "start"})
        )
        self._publish_status()
        self.get_logger().info(f"recording started: {self.active_run_dir}")

    def _finish_run(self) -> None:
        if not self.recording:
            self.get_logger().warning("record finish ignored: not recording")
            return
        self.record_finish_recv_ns = self._recv_ns()
        self.record_control_rows.append(
            TopicRow(stamp_ns=0, recv_ns=self.record_finish_recv_ns, payload={"command": 2, "label": "finish"})
        )
        self.flush()
        self.recording = False
        self.active_run_dir = None
        self._publish_status()
        self.get_logger().info("recording finished and saved")

    def on_ready(self, msg: Bool) -> None:
        self.latest_ready = bool(msg.data)

    def on_episode_meta(self, msg: String) -> None:
        try:
            self.latest_episode_meta = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            self.latest_episode_meta = {"raw": msg.data}

    def on_record_control(self, msg: Int32) -> None:
        cmd = int(msg.data)
        if cmd == 1:
            self._begin_run()
        elif cmd == 2:
            self._finish_run()

    def _write_image(self, msg: Image, index: int, subdir: str) -> str:
        assert self.active_run_dir is not None
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

        rel = f"{subdir}/{index:06d}.npy"
        np.save(self.active_run_dir / rel, img)
        return rel

    def _write_lidar(self, msg: PointCloud2, index: int) -> str:
        assert self.active_run_dir is not None
        pts = _pointcloud_xyz_array(msg)
        rel = f"lidar/{index:06d}.npy"
        np.save(self.active_run_dir / rel, pts)
        return rel

    def _write_table(self, df: pd.DataFrame, stem: str) -> str:
        assert self.active_run_dir is not None
        if self.table_format == "parquet":
            try:
                path = self.active_run_dir / f"{stem}.parquet"
                df.to_parquet(path, index=False)
                return path.name
            except ImportError:
                self.table_format = "pickle"
                self.get_logger().warning("pyarrow/fastparquet not found; falling back to pickle tables")
        path = self.active_run_dir / f"{stem}.pkl"
        df.to_pickle(path)
        return path.name

    def on_driver_rgb(self, msg: Image) -> None:
        if not self.recording:
            return
        idx = len(self.driver_rows)
        rel_path = self._write_image(msg, idx, "camera_driver")
        self.driver_rows.append(
            TopicRow(self._stamp_ns(msg), self._recv_ns(), {"path": rel_path, "encoding": msg.encoding})
        )
        self._tick()

    def on_bucket_rgb(self, msg: Image) -> None:
        if not self.recording:
            return
        idx = len(self.bucket_rows)
        rel_path = self._write_image(msg, idx, "camera_bucket")
        self.bucket_rows.append(
            TopicRow(self._stamp_ns(msg), self._recv_ns(), {"path": rel_path, "encoding": msg.encoding})
        )
        self._tick()

    def on_lidar(self, msg: PointCloud2) -> None:
        if not self.recording:
            return
        idx = len(self.lidar_rows)
        rel_path = self._write_lidar(msg, idx)
        self.lidar_rows.append(
            TopicRow(self._stamp_ns(msg), self._recv_ns(), {"path": rel_path, "frame_id": msg.header.frame_id})
        )
        self._tick()

    def on_joint(self, msg: JointState) -> None:
        if not self.recording:
            return
        self.proprio_rows.append(
            TopicRow(
                self._stamp_ns(msg),
                self._recv_ns(),
                {
                    "name": list(msg.name),
                    "position": list(msg.position),
                    "velocity": list(msg.velocity),
                    "effort": list(msg.effort),
                },
            )
        )
        self._tick()

    def on_cmd(self, msg: JointState) -> None:
        if not self.recording:
            return
        self.action_rows.append(
            TopicRow(
                self._stamp_ns(msg),
                self._recv_ns(),
                {
                    "name": list(msg.name),
                    "position": list(msg.position),
                    "velocity": list(msg.velocity),
                    "effort": list(msg.effort),
                },
            )
        )
        self._tick()

    def on_stones_in_truck(self, msg: Int32) -> None:
        if not self.recording:
            return
        self.stones_rows.append(
            TopicRow(
                self._stamp_ns(msg),
                self._recv_ns(),
                {
                    "count": int(msg.data),
                },
            )
        )
        self._tick()

    def _tick(self):
        self._counter += 1
        now_ns = self._recv_ns()
        if now_ns - self._last_log_ns > 1_000_000_000:
            self._publish_status()
            self.get_logger().info(
                "driver=%d bucket=%d lidar=%d proprio=%d action=%d"
                % (
                    len(self.driver_rows),
                    len(self.bucket_rows),
                    len(self.lidar_rows),
                    len(self.proprio_rows),
                    len(self.action_rows),
                )
            )
            self._last_log_ns = now_ns

    def _publish_status(self) -> None:
        run_id = self.active_run_dir.name if self.active_run_dir is not None else ""
        msg = String()
        msg.data = json.dumps(
            {
                "recording": self.recording,
                "run_id": run_id,
                "counts": {
                    "camera_driver": len(self.driver_rows),
                    "camera_bucket": len(self.bucket_rows),
                    "lidar": len(self.lidar_rows),
                    "joint_states": len(self.proprio_rows),
                    "cmd_joint": len(self.action_rows),
                },
            }
        )
        self.status_publisher.publish(msg)

    @staticmethod
    def _rows_to_df(rows: List[TopicRow], prefix: str) -> pd.DataFrame:
        data = []
        for r in rows:
            item = {"stamp_ns": r.stamp_ns, f"{prefix}_recv_ns": r.recv_ns}
            for k, v in r.payload.items():
                item[f"{prefix}_{k}"] = v
            data.append(item)
        if not data:
            return pd.DataFrame(columns=["stamp_ns", f"{prefix}_recv_ns"])
        return pd.DataFrame(data).sort_values(["stamp_ns", f"{prefix}_recv_ns"]).reset_index(drop=True)

    @staticmethod
    def _average_hz_from_ns(values_ns: List[int]) -> float:
        if len(values_ns) < 2:
            return 0.0
        values = np.asarray([int(v) for v in values_ns if int(v) > 0], dtype=np.int64)
        if values.size < 2:
            return 0.0
        diffs = np.diff(np.sort(values)) / 1e9
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return 0.0
        return float(1.0 / diffs.mean())

    @classmethod
    def _average_hz_recv(cls, rows: List[TopicRow]) -> float:
        if len(rows) < 2:
            return 0.0
        return cls._average_hz_from_ns([r.recv_ns for r in rows])

    @classmethod
    def _average_hz_stamp(cls, rows: List[TopicRow]) -> float:
        if len(rows) < 2:
            return 0.0
        return cls._average_hz_from_ns([r.stamp_ns for r in rows])

    def flush(self):
        if self.active_run_dir is None:
            return

        driver_df = self._rows_to_df(self.driver_rows, "driver")
        bucket_df = self._rows_to_df(self.bucket_rows, "bucket")
        lidar_df = self._rows_to_df(self.lidar_rows, "lidar")
        proprio_df = self._rows_to_df(self.proprio_rows, "proprio")
        action_df = self._rows_to_df(self.action_rows, "action")
        stones_df = self._rows_to_df(self.stones_rows, "stones")
        record_ctrl_df = self._rows_to_df(self.record_control_rows, "record")

        all_recv_ns: List[int] = []
        all_stamp_ns: List[int] = []
        for rows in (self.driver_rows, self.bucket_rows, self.lidar_rows, self.proprio_rows, self.action_rows, self.stones_rows):
            all_recv_ns.extend([r.recv_ns for r in rows if r.recv_ns > 0])
            all_stamp_ns.extend([r.stamp_ns for r in rows if r.stamp_ns > 0])

        total_episode_time_s = 0.0
        if self.record_start_recv_ns is not None and self.record_finish_recv_ns is not None:
            total_episode_time_s = float((self.record_finish_recv_ns - self.record_start_recv_ns) / 1e9)
        elif all_recv_ns:
            total_episode_time_s = float((max(all_recv_ns) - min(all_recv_ns)) / 1e9)

        ros_stamp_span_s = 0.0
        if all_stamp_ns:
            ros_stamp_span_s = float((max(all_stamp_ns) - min(all_stamp_ns)) / 1e9)

        meta = {
            "topics": {
                "camera_driver": "/excavator/camera_driver/rgb",
                "camera_bucket": "/excavator/camera_bucket/rgb",
                "lidar": "/excavator/lidar/points",
                "joint_states": "/excavator/joint_states",
                "cmd_joint": "/excavator/cmd_joint",
                "stones_in_truck": "/excavator/stones_in_truck",
                "record_control": "/excavator/record_control",
                "ready": "/excavator/ready",
                "episode_meta": "/excavator/episode_meta",
            },
            "episode_meta": self.latest_episode_meta,
            "record_window": {
                "start_recv_ns": self.record_start_recv_ns,
                "finish_recv_ns": self.record_finish_recv_ns,
                "ready_at_finish": self.latest_ready,
            },
            "counts": {
                "camera_driver": len(driver_df),
                "camera_bucket": len(bucket_df),
                "lidar": len(lidar_df),
                "proprio": len(proprio_df),
                "action": len(action_df),
                "stones_in_truck": len(stones_df),
            },
            "topic_hz_avg_recv": {
                "camera_driver": self._average_hz_recv(self.driver_rows),
                "camera_bucket": self._average_hz_recv(self.bucket_rows),
                "lidar": self._average_hz_recv(self.lidar_rows),
                "joint_states": self._average_hz_recv(self.proprio_rows),
                "cmd_joint": self._average_hz_recv(self.action_rows),
                "stones_in_truck": self._average_hz_recv(self.stones_rows),
            },
            "topic_hz_avg_ros_stamp": {
                "camera_driver": self._average_hz_stamp(self.driver_rows),
                "camera_bucket": self._average_hz_stamp(self.bucket_rows),
                "lidar": self._average_hz_stamp(self.lidar_rows),
                "joint_states": self._average_hz_stamp(self.proprio_rows),
                "cmd_joint": self._average_hz_stamp(self.action_rows),
                "stones_in_truck": self._average_hz_stamp(self.stones_rows),
            },
            "total_episode_time_s": total_episode_time_s,
            "ros_stamp_span_s": ros_stamp_span_s,
        }

        self._write_table(driver_df, "camera_driver")
        self._write_table(bucket_df, "camera_bucket")
        self._write_table(lidar_df, "lidar")
        self._write_table(proprio_df, "proprio")
        self._write_table(action_df, "action")
        self._write_table(stones_df, "stones_in_truck")
        self._write_table(record_ctrl_df, "record_control")
        meta["table_format"] = self.table_format
        (self.active_run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.get_logger().info(f"saved run: {self.active_run_dir}")
        self._publish_status()


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
    parser = argparse.ArgumentParser(description="Record excavator ROS2 topics by control topic")
    parser.add_argument("--out-dir", default="", help="Single run directory. If set, first start uses this path.")
    parser.add_argument("--base-dir", default=str(paths.raw_data), help="Base directory for run_xxx")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    node = Recorder(base_dir=Path(args.base_dir), out_dir=out_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording:
            node.flush()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
