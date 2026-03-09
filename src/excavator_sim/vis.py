from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool


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
        self.current_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.target_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.left_image: Optional[pygame.Surface] = None
        self.right_image: Optional[pygame.Surface] = None

        self.create_subscription(Bool, "/excavator/ready", self._on_ready, 10)
        self.create_subscription(JointState, "/excavator/joint_states", self._on_joint_states, 50)
        self.create_subscription(JointState, "/excavator/cmd_joint", self._on_cmd_joint, 50)
        self.create_subscription(Image, "/excavator/camera_front_left/rgb", self._on_left_image, 10)
        self.create_subscription(Image, "/excavator/camera_front_right/rgb", self._on_right_image, 10)

    def _on_ready(self, msg: Bool) -> None:
        self.ready = bool(msg.data)

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

    def _on_left_image(self, msg: Image) -> None:
        surf = self._msg_to_surface(msg)
        if surf is not None:
            self.left_image = surf

    def _on_right_image(self, msg: Image) -> None:
        surf = self._msg_to_surface(msg)
        if surf is not None:
            self.right_image = surf


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
            left_rect = pygame.Rect(40, 40, panel_w, panel_h)
            right_rect = pygame.Rect(720, 40, panel_w, panel_h)
            pygame.draw.rect(screen, (50, 50, 50), left_rect, width=2)
            pygame.draw.rect(screen, (50, 50, 50), right_rect, width=2)

            if node.left_image is not None:
                screen.blit(pygame.transform.smoothscale(node.left_image, (panel_w, panel_h)), left_rect.topleft)
            else:
                screen.blit(small_font.render("left camera: no image", True, (200, 100, 100)), (left_rect.x + 16, left_rect.y + 16))
            if node.right_image is not None:
                screen.blit(pygame.transform.smoothscale(node.right_image, (panel_w, panel_h)), right_rect.topleft)
            else:
                screen.blit(small_font.render("right camera: no image", True, (200, 100, 100)), (right_rect.x + 16, right_rect.y + 16))

            state_color = (50, 200, 80) if node.ready else (220, 70, 70)
            state_text = "state: ready" if node.ready else "state: not ready"
            screen.blit(font.render(state_text, True, state_color), (40, 430))

            y = 470
            for j in JOINT_ORDER:
                cp = node.current_positions[j]
                tp = node.target_positions[j]
                line = f"{j:12s} cur={cp:+.3f} rad  tgt={tp:+.3f} rad"
                screen.blit(small_font.render(line, True, (220, 220, 220)), (40, y))
                y += 34

            pygame.display.flip()
            clock.tick(cfg.hz)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        pygame.quit()


if __name__ == "__main__":
    main()

