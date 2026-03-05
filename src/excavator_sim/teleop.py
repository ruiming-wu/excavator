from __future__ import annotations

import argparse
import sys
import termios
import tty
from dataclasses import dataclass
from typing import Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


@dataclass
class TeleopConfig:
    joint_names: List[str]
    step: float
    mode: str  # position | velocity


HELP = """
Excavator teleop keys:
  q/a: boom_up/down
  w/s: arm_up/down
  e/d: bucket_up/down
  r/f: swing_left/right
  t/g: base_forward/backward
  z:   zero all
  x:   quit
""".strip()


class TeleopNode(Node):
    def __init__(self, cfg: TeleopConfig):
        super().__init__("excavator_teleop")
        self.cfg = cfg
        self.publisher = self.create_publisher(JointState, "/excavator/cmd_joint", 10)
        self.values: Dict[str, float] = {j: 0.0 for j in cfg.joint_names}

        self.keymap = {
            "q": ("boom_joint", +cfg.step),
            "a": ("boom_joint", -cfg.step),
            "w": ("arm_joint", +cfg.step),
            "s": ("arm_joint", -cfg.step),
            "e": ("bucket_joint", +cfg.step),
            "d": ("bucket_joint", -cfg.step),
            "r": ("swing_joint", +cfg.step),
            "f": ("swing_joint", -cfg.step),
            "t": ("track_joint", +cfg.step),
            "g": ("track_joint", -cfg.step),
        }

    def apply_key(self, key: str) -> bool:
        if key == "z":
            for k in self.values:
                self.values[k] = 0.0
            self.publish_command()
            return True
        if key == "x":
            return False
        if key in self.keymap:
            name, delta = self.keymap[key]
            self.values[name] += delta
            self.publish_command()
        return True

    def publish_command(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.values.keys())
        if self.cfg.mode == "position":
            msg.position = [self.values[n] for n in msg.name]
            msg.velocity = []
        else:
            msg.velocity = [self.values[n] for n in msg.name]
            msg.position = []
        self.publisher.publish(msg)


class RawKeyboard:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read_key(self) -> str:
        return sys.stdin.read(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard teleop for excavator joints")
    parser.add_argument("--mode", choices=["position", "velocity"], default="position")
    parser.add_argument("--step", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    cfg = TeleopConfig(
        joint_names=["boom_joint", "arm_joint", "bucket_joint", "swing_joint", "track_joint"],
        step=args.step,
        mode=args.mode,
    )
    node = TeleopNode(cfg)
    print(HELP)
    try:
        with RawKeyboard() as kb:
            running = True
            while running and rclpy.ok():
                key = kb.read_key()
                running = node.apply_key(key)
                rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
