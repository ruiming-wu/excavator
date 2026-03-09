from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Int32


JOINT_ORDER = ["boom_joint", "arm_joint", "bucket_joint", "swing_joint"]


@dataclass
class VisConfig:
    hz: float
    width: int
    height: int


class ExcavatorVisNode(Node):
    def __init__(self):
        super().__init__("excavator_vis")
        self.ready = False
        self.stones_in_truck = 0
        self.current_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.target_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.driver_image: Optional[pygame.Surface] = None
        self.bucket_image: Optional[pygame.Surface] = None
        self.lidar_xy: Optional[np.ndarray] = None

        self.create_subscription(Bool, "/excavator/ready", self._on_ready, 10)
        self.create_subscription(Int32, "/excavator/stones_in_truck", self._on_stones_in_truck, 10)
        self.create_subscription(JointState, "/excavator/joint_states", self._on_joint_states, 50)
        self.create_subscription(JointState, "/excavator/cmd_joint", self._on_cmd_joint, 50)
        self.create_subscription(Image, "/excavator/camera_driver/rgb", self._on_driver_image, 10)
        self.create_subscription(Image, "/excavator/camera_bucket/rgb", self._on_bucket_image, 10)
        self.create_subscription(PointCloud2, "/excavator/lidar/points", self._on_lidar, 10)

    def _on_ready(self, msg: Bool) -> None:
        self.ready = bool(msg.data)

    def _on_stones_in_truck(self, msg: Int32) -> None:
        self.stones_in_truck = int(msg.data)

    def _on_joint_states(self, msg: JointState) -> None:
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        for joint in JOINT_ORDER:
            if joint in name_to_pos:
                self.current_positions[joint] = float(name_to_pos[joint])

    def _on_cmd_joint(self, msg: JointState) -> None:
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        for joint in JOINT_ORDER:
            if joint in name_to_pos:
                self.target_positions[joint] = float(name_to_pos[joint])

    @staticmethod
    def _msg_to_surface(msg: Image) -> Optional[pygame.Surface]:
        w = int(msg.width)
        h = int(msg.height)
        step = int(msg.step)
        if w <= 0 or h <= 0 or step <= 0:
            return None
        enc = str(msg.encoding).lower()
        channels = step // w if w > 0 else 0
        if channels < 3:
            return None

        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if arr.size < h * step:
            return None
        arr = arr[: h * step].reshape((h, step))
        arr = arr[:, : w * channels].reshape((h, w, channels))

        if enc in ("rgb8", "rgba8"):
            rgb = arr[:, :, :3]
        elif enc in ("bgr8", "bgra8"):
            rgb = arr[:, :, :3][:, :, ::-1]
        else:
            return None

        rgb_whc = np.transpose(rgb, (1, 0, 2)).copy()
        return pygame.surfarray.make_surface(rgb_whc)

    def _on_driver_image(self, msg: Image) -> None:
        surf = self._msg_to_surface(msg)
        if surf is not None:
            self.driver_image = surf

    def _on_bucket_image(self, msg: Image) -> None:
        surf = self._msg_to_surface(msg)
        if surf is not None:
            self.bucket_image = surf

    def _on_lidar(self, msg: PointCloud2) -> None:
        # Keep a capped point set for responsive UI.
        pts_iter = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = list(itertools.islice(pts_iter, 6000))
        if not pts:
            self.lidar_xy = None
            return
        first = pts[0]
        # read_points may return tuples or structured arrays depending on backend.
        if hasattr(first, "dtype") and getattr(first.dtype, "names", None):
            x = np.asarray([float(p["x"]) for p in pts], dtype=np.float32)
            y = np.asarray([float(p["y"]) for p in pts], dtype=np.float32)
        else:
            arr = np.asarray(pts, dtype=np.float32)
            x = arr[:, 0]
            y = arr[:, 1]
        # Use top-down XY projection (x forward, y left/right).
        self.lidar_xy = np.stack([x, y], axis=1)


def parse_args() -> VisConfig:
    parser = argparse.ArgumentParser(description="Excavator visualization UI")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=860)
    args = parser.parse_args()
    return VisConfig(hz=args.hz, width=args.width, height=args.height)


def main() -> None:
    cfg = parse_args()
    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption("Excavator Visualization")
    font = pygame.font.SysFont("monospace", 24)
    small_font = pygame.font.SysFont("monospace", 20)

    rclpy.init()
    node = ExcavatorVisNode()
    clock = pygame.time.Clock()

    try:
        while rclpy.ok():
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    quit_requested = True
            if quit_requested:
                break

            rclpy.spin_once(node, timeout_sec=0.0)

            screen.fill((24, 24, 24))
            panel_w, panel_h = 640, 360
            driver_rect = pygame.Rect(40, 40, panel_w, panel_h)
            bucket_rect = pygame.Rect(720, 40, panel_w, panel_h)
            pygame.draw.rect(screen, (50, 50, 50), driver_rect, width=2)
            pygame.draw.rect(screen, (50, 50, 50), bucket_rect, width=2)

            if node.driver_image is not None:
                screen.blit(pygame.transform.smoothscale(node.driver_image, (panel_w, panel_h)), driver_rect.topleft)
            else:
                screen.blit(small_font.render("driver camera: no image", True, (200, 100, 100)), (driver_rect.x + 16, driver_rect.y + 16))
            if node.bucket_image is not None:
                screen.blit(pygame.transform.smoothscale(node.bucket_image, (panel_w, panel_h)), bucket_rect.topleft)
            else:
                screen.blit(small_font.render("bucket camera: no image", True, (200, 100, 100)), (bucket_rect.x + 16, bucket_rect.y + 16))

            lidar_rect = pygame.Rect(760, 430, 560, 380)
            pygame.draw.rect(screen, (50, 50, 50), lidar_rect, width=2)
            screen.blit(small_font.render("lidar bev", True, (180, 180, 180)), (lidar_rect.x + 12, lidar_rect.y + 8))
            if node.lidar_xy is not None and node.lidar_xy.size > 0:
                x_m = node.lidar_xy[:, 0]
                y_m = node.lidar_xy[:, 1]
                # Crop to a practical local window in meters.
                mask = (x_m >= 0.0) & (x_m <= 10.0) & (y_m >= -5.0) & (y_m <= 5.0)
                x_m = x_m[mask]
                y_m = y_m[mask]
                if x_m.size > 0:
                    # Map meters -> pixels: forward to up, left to left.
                    px = lidar_rect.x + ((5.0 - y_m) / 10.0) * lidar_rect.width
                    py = lidar_rect.y + (1.0 - (x_m / 10.0)) * lidar_rect.height
                    pts_px = np.stack([px, py], axis=1).astype(np.int32)
                    for p in pts_px:
                        pygame.draw.circle(screen, (110, 210, 255), p, 1)
            else:
                screen.blit(small_font.render("lidar: no points", True, (200, 100, 100)), (lidar_rect.x + 16, lidar_rect.y + 36))

            state_color = (50, 200, 80) if node.ready else (220, 70, 70)
            state_text = "state: ready" if node.ready else "state: not ready"
            screen.blit(font.render(state_text, True, state_color), (40, 430))
            stones_text = f"stones in truck: {node.stones_in_truck}"
            screen.blit(font.render(stones_text, True, (220, 220, 220)), (40, 465))

            y = 510
            for j in JOINT_ORDER:
                cp = node.current_positions[j]
                tp = node.target_positions[j]
                line = f"{j:12s} cur={cp:+.3f} rad  tgt={tp:+.3f} rad"
                screen.blit(small_font.render(line, True, (220, 220, 220)), (40, y))
                y += 34

            hint_y = y + 12
            screen.blit(small_font.render("reset env: key 3 / gamepad X", True, (180, 180, 180)), (40, hint_y))
            screen.blit(small_font.render("reset joints to zero: key 4 / gamepad Y", True, (180, 180, 180)), (40, hint_y + 28))

            pygame.display.flip()
            clock.tick(cfg.hz)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        pygame.quit()


if __name__ == "__main__":
    main()
