from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_msgs.msg import Int32


JOINT_ORDER = ["boom_joint", "arm_joint", "bucket_joint", "swing_joint"]


@dataclass
class TeleopConfig:
    urdf_path: Path
    scale: float
    hz: float
    deadzone: float
    method: str
    joystick_index: int
    left_x_axis: int
    left_y_axis: int
    right_x_axis: int
    right_y_axis: int
    x_button: int
    y_button: int
    zero_goal_tol: float


class JoystickTeleopNode(Node):
    def __init__(self, cfg: TeleopConfig):
        super().__init__("excavator_joystick_teleop")
        self.cfg = cfg
        self.publisher = self.create_publisher(JointState, "/excavator/cmd_joint", 50)
        self.reset_publisher = self.create_publisher(Int32, "/excavator/reset", 10)
        self.subscriber = self.create_subscription(JointState, "/excavator/joint_states", self.on_joint_states, 50)
        self.ready_subscriber = self.create_subscription(Bool, "/excavator/ready", self.on_ready, 10)

        self.current_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.target_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.has_joint_state = False
        self.ready = False
        self.zero_goal_blocking = False

        self.joint_limits = self._load_joint_limits(cfg.urdf_path)
        self.get_logger().info(f"Loaded joint limits from {cfg.urdf_path}")

    def request_env_reset(self) -> None:
        msg = Int32()
        msg.data = 1
        self.reset_publisher.publish(msg)
        self.get_logger().info("Published /excavator/reset = 1")

    def reset_joint_targets_to_zero(self) -> None:
        self._start_zero_goal_blocking()
        self.get_logger().info("Y pressed: keep sending zero until joints are near zero")

    def _start_zero_goal_blocking(self) -> None:
        self.target_positions = {j: 0.0 for j in JOINT_ORDER}
        self.zero_goal_blocking = True
        self._publish_zero_joint_cmd_once()

    def _publish_zero_joint_cmd_once(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_ORDER
        msg.position = [0.0 for _ in JOINT_ORDER]
        self.publisher.publish(msg)

    def _is_joint_state_close_to_zero(self) -> bool:
        tol = float(self.cfg.zero_goal_tol)
        return all(abs(float(self.current_positions[j])) <= tol for j in JOINT_ORDER)

    def _load_joint_limits(self, urdf_path: Path) -> Dict[str, tuple[float, float]]:
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        tree = ET.parse(urdf_path)
        root = tree.getroot()
        limits: Dict[str, tuple[float, float]] = {}

        for joint in root.findall("joint"):
            name = joint.get("name", "")
            if name not in JOINT_ORDER:
                continue
            limit = joint.find("limit")
            if limit is None:
                continue
            lower = float(limit.get("lower", "-3.1415926"))
            upper = float(limit.get("upper", "3.1415926"))
            limits[name] = (lower, upper)

        missing = [j for j in JOINT_ORDER if j not in limits]
        if missing:
            raise RuntimeError(f"Missing joint limits in URDF: {missing}")

        return limits

    def on_joint_states(self, msg: JointState) -> None:
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        changed = False
        for joint in JOINT_ORDER:
            if joint in name_to_pos:
                self.current_positions[joint] = float(name_to_pos[joint])
                changed = True

        if not changed:
            return

        if not self.has_joint_state:
            self.target_positions = dict(self.current_positions)
            self.has_joint_state = True
            self.get_logger().info("Received first /excavator/joint_states; teleop enabled")

    def on_ready(self, msg: Bool) -> None:
        new_state = bool(msg.data)
        if self.ready != new_state:
            self.ready = new_state
            self.get_logger().info(f"Ready state changed: {self.ready}")
            if not self.ready:
                # Stop any ongoing forced-zero operation during reset/init.
                self.zero_goal_blocking = False

    @staticmethod
    def _apply_deadzone(v: float, dz: float) -> float:
        return 0.0 if abs(v) < dz else v

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def step(self, left_x: float, left_y: float, right_x: float, right_y: float) -> None:
        if not self.ready:
            return
        if self.zero_goal_blocking:
            self._publish_zero_joint_cmd_once()
            if self.has_joint_state and self._is_joint_state_close_to_zero():
                self.zero_goal_blocking = False
                self.target_positions = dict(self.current_positions)
                self.get_logger().info("Zero-goal reached; exit blocking mode")
            return

        if not self.has_joint_state:
            return

        lx = self._apply_deadzone(left_x, self.cfg.deadzone)
        ly = self._apply_deadzone(left_y, self.cfg.deadzone)
        rx = self._apply_deadzone(right_x, self.cfg.deadzone)
        ry = self._apply_deadzone(right_y, self.cfg.deadzone)

        # Mapping requested by user (all values are in radians):
        # 1) Left stick up/down -> arm, up smaller, down larger: delta = +scale * ly
        # 2) Left stick left/right -> swing, left larger, right smaller: delta = -scale * lx
        # 3) Right stick up/down -> boom, up larger: delta = -scale * ry
        # 4) Right stick left/right -> bucket, left larger: delta = -scale * rx
        updates = {
            "arm_joint": self.target_positions["arm_joint"] + self.cfg.scale * ly,
            "swing_joint": self.target_positions["swing_joint"] - self.cfg.scale * lx,
            "boom_joint": self.target_positions["boom_joint"] - self.cfg.scale * ry,
            "bucket_joint": self.target_positions["bucket_joint"] - self.cfg.scale * rx,
        }

        for joint, new_target in updates.items():
            lo, hi = self.joint_limits[joint]
            self.target_positions[joint] = self._clamp(new_target, lo, hi)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_ORDER
        msg.position = [self.target_positions[j] for j in JOINT_ORDER]
        self.publisher.publish(msg)


def _default_urdf_path() -> Path:
    env_path = os.environ.get("EXCAVATOR_ASSET_PATH", "").strip()
    if env_path and env_path.endswith(".urdf"):
        return Path(env_path)
    return Path.cwd() / "assets" / "excavator" / "excavator_4dof.urdf"


def parse_args() -> TeleopConfig:
    parser = argparse.ArgumentParser(description="Joystick-only teleop for excavator (target joint positions)")
    parser.add_argument("--urdf", type=str, default=str(_default_urdf_path()))
    parser.add_argument("--scale", type=float, default=0.02, help="Per-step delta: target = current + scale * joystick")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument("--method", type=str, choices=["joystick", "keyboard"], default="joystick")
    parser.add_argument("--joystick-index", type=int, default=0)

    # Axis defaults for many Xbox mappings in pygame.
    parser.add_argument("--left-x-axis", type=int, default=0)
    parser.add_argument("--left-y-axis", type=int, default=1)
    parser.add_argument("--right-x-axis", type=int, default=2)
    parser.add_argument("--right-y-axis", type=int, default=3)
    parser.add_argument("--x-button", type=int, default=3, help="Xbox X button index in pygame")
    parser.add_argument("--y-button", type=int, default=4, help="Xbox Y button index in pygame")
    parser.add_argument("--zero-goal-tol", type=float, default=0.05, help="Close-to-zero tolerance in radians")

    args = parser.parse_args()
    return TeleopConfig(
        urdf_path=Path(args.urdf),
        scale=args.scale,
        hz=args.hz,
        deadzone=args.deadzone,
        method=args.method,
        joystick_index=args.joystick_index,
        left_x_axis=args.left_x_axis,
        left_y_axis=args.left_y_axis,
        right_x_axis=args.right_x_axis,
        right_y_axis=args.right_y_axis,
        x_button=args.x_button,
        y_button=args.y_button,
        zero_goal_tol=args.zero_goal_tol,
    )


def _keyboard_to_controls() -> tuple[float, float, float, float, bool, bool]:
    keys = pygame.key.get_pressed()
    lx = 0.0
    ly = 0.0
    rx = 0.0
    ry = 0.0

    if keys[pygame.K_a]:
        lx -= 1.0
    if keys[pygame.K_d]:
        lx += 1.0
    if keys[pygame.K_w]:
        ly -= 1.0
    if keys[pygame.K_s]:
        ly += 1.0

    if keys[pygame.K_j]:
        rx -= 1.0
    if keys[pygame.K_l]:
        rx += 1.0
    if keys[pygame.K_i]:
        ry -= 1.0
    if keys[pygame.K_k]:
        ry += 1.0

    x_pressed = bool(keys[pygame.K_3])
    y_pressed = bool(keys[pygame.K_4])
    return lx, ly, rx, ry, x_pressed, y_pressed


def main() -> None:
    cfg = parse_args()

    pygame.init()
    js = None

    if cfg.method == "joystick":
        pygame.joystick.init()
        if pygame.joystick.get_count() <= cfg.joystick_index:
            print(
                f"No joystick at index {cfg.joystick_index}. connected={pygame.joystick.get_count()}",
                file=sys.stderr,
            )
            sys.exit(1)

        js = pygame.joystick.Joystick(cfg.joystick_index)
        js.init()
        print(f"[teleop] input method: joystick")
        print(f"[teleop] joystick: {js.get_name()} (index={cfg.joystick_index})")
        print("[teleop] controls:")
        print("  left Y  -> arm (up smaller, down larger)")
        print("  left X  -> swing (left larger, right smaller)")
        print("  right Y -> boom (up larger)")
        print("  right X -> bucket (left larger)")
        print("  X button -> env reset only")
        print("  Y button -> zero-goal blocking")
        print("Ctrl+C to quit")
    else:
        pygame.display.init()
        pygame.display.set_mode((480, 120))
        pygame.display.set_caption("Excavator Teleop Keyboard")
        print("[teleop] input method: keyboard")
        print("[teleop] controls:")
        print("  WASD -> left stick (A/D: X, W/S: Y)")
        print("  IJKL -> right stick (J/L: X, I/K: Y)")
        print("  3 -> X button (env reset)")
        print("  4 -> Y button (zero-goal blocking)")
        print("Keep the pygame window focused. Ctrl+C to quit")

    rclpy.init()
    node = JoystickTeleopNode(cfg)
    period = 1.0 / cfg.hz
    prev_x_pressed = False
    prev_y_pressed = False

    try:
        while rclpy.ok():
            pygame.event.pump()
            if cfg.method == "joystick":
                lx = float(js.get_axis(cfg.left_x_axis))
                ly = float(js.get_axis(cfg.left_y_axis))
                rx = float(js.get_axis(cfg.right_x_axis))
                ry = float(js.get_axis(cfg.right_y_axis))
                x_pressed = bool(js.get_button(cfg.x_button))
                y_pressed = bool(js.get_button(cfg.y_button))
            else:
                lx, ly, rx, ry, x_pressed, y_pressed = _keyboard_to_controls()

            rclpy.spin_once(node, timeout_sec=0.0)
            if x_pressed and not prev_x_pressed:
                node.request_env_reset()
            elif y_pressed and not prev_y_pressed and node.ready:
                node.reset_joint_targets_to_zero()
            else:
                node.step(lx, ly, rx, ry)
            prev_x_pressed = x_pressed
            prev_y_pressed = y_pressed
            time.sleep(period)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if js is not None:
            js.quit()
        if cfg.method == "keyboard":
            pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
