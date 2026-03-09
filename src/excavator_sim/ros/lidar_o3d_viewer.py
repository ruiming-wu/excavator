from __future__ import annotations

import argparse
import itertools
from typing import Optional

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class LidarO3DViewer(Node):
    def __init__(self, topic: str, max_points: int):
        super().__init__("lidar_o3d_viewer")
        self.topic = topic
        self.max_points = max(1, int(max_points))
        self._latest_points: Optional[np.ndarray] = None
        self._updated = False
        self._msg_count = 0
        self._last_stats_log = 0

        self.create_subscription(
            PointCloud2,
            self.topic,
            self._on_lidar,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"subscribing: {self.topic} (qos=sensor_data, max_points={self.max_points})")

    def _on_lidar(self, msg: PointCloud2) -> None:
        pts_iter = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = list(itertools.islice(pts_iter, self.max_points))
        self._msg_count += 1

        if not pts:
            self._latest_points = None
            self._updated = True
            return

        first = pts[0]
        # Different backends may return tuple-like or structured element.
        if hasattr(first, "dtype") and getattr(first.dtype, "names", None):
            x = np.asarray([float(p["x"]) for p in pts], dtype=np.float32)
            y = np.asarray([float(p["y"]) for p in pts], dtype=np.float32)
            z = np.asarray([float(p["z"]) for p in pts], dtype=np.float32)
            arr = np.stack([x, y, z], axis=1)
        else:
            arr = np.asarray(pts, dtype=np.float32)

        finite_mask = np.isfinite(arr).all(axis=1)
        arr = arr[finite_mask]
        if arr.size == 0:
            self._latest_points = None
            self._updated = True
            return

        self._latest_points = arr
        self._updated = True

        if self._msg_count % 30 == 0:
            self.get_logger().info(f"received cloud msgs={self._msg_count}, points={arr.shape[0]}")
        if self._msg_count - self._last_stats_log >= 120:
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            self.get_logger().info(
                "xyz range: "
                f"x=[{mins[0]:.3f},{maxs[0]:.3f}] "
                f"y=[{mins[1]:.3f},{maxs[1]:.3f}] "
                f"z=[{mins[2]:.3f},{maxs[2]:.3f}]"
            )
            self._last_stats_log = self._msg_count

    def consume_points(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._updated:
            return False, None
        self._updated = False
        return True, self._latest_points


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal ROS2 PointCloud2 -> Open3D viewer")
    parser.add_argument("--topic", default="/excavator/lidar/points")
    parser.add_argument("--max-points", type=int, default=120000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = LidarO3DViewer(topic=args.topic, max_points=args.max_points)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Lidar Open3D Viewer", width=1280, height=720)
    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = 1.5
        render_opt.background_color = np.array([0.02, 0.02, 0.02], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(axes)
    view_initialized = False
    view_reset_count = 0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            updated, points = node.consume_points()
            if updated:
                if points is None or points.size == 0:
                    pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
                    pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
                else:
                    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                    # Use a bright fixed color for visibility.
                    colors = np.tile(np.array([[0.1, 0.9, 0.4]], dtype=np.float64), (points.shape[0], 1))
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(pcd)
                if points is not None and points.size > 0 and (not view_initialized or view_reset_count < 5):
                    bbox = pcd.get_axis_aligned_bounding_box()
                    center = bbox.get_center()
                    vc = vis.get_view_control()
                    vc.set_lookat(center)
                    vc.set_front(np.array([0.2, -1.0, -0.35], dtype=np.float64))
                    vc.set_up(np.array([0.0, 0.0, 1.0], dtype=np.float64))
                    vc.set_zoom(0.35)
                    view_initialized = True
                    view_reset_count += 1

            vis.poll_events()
            vis.update_renderer()
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
