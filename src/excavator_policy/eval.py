from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pygame
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Int32, String

from excavator_policy.dataset import _named_vector
from excavator_policy.model import DiffusionPolicy
from excavator_sim.common import get_paths


@dataclass
class EvalConfig:
    checkpoint: Path
    episodes: int
    max_seconds: float
    control_hz: float
    sample_steps: int
    startup_timeout: float
    reset_timeout: float
    output_dir: Path | None
    euler_step_size: float
    success_hold_seconds: float
    record_video: bool
    record_display: str
    record_fps: int


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate policy in a live Isaac Sim ROS environment")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--reset-timeout", type=float, default=20.0)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--euler-step-size", type=float, default=0.1, help="Euler integration step size for flow matching decoding")
    parser.add_argument("--success-hold-seconds", type=float, default=5.0, help="End episode early and mark success once stones_in_truck stays > 0 for this many seconds")
    parser.add_argument("--record-video", action="store_true", help="Record each episode to an mp4 via ffmpeg x11 capture")
    parser.add_argument("--record-display", type=str, default=os.environ.get("DISPLAY", ":0"), help="Display target for ffmpeg x11 capture")
    parser.add_argument("--record-fps", type=int, default=30, help="Recording frame rate when --record-video is enabled")
    args = parser.parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    return EvalConfig(
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        episodes=int(args.episodes),
        max_seconds=float(args.max_seconds),
        control_hz=float(args.control_hz),
        sample_steps=int(args.sample_steps),
        startup_timeout=float(args.startup_timeout),
        reset_timeout=float(args.reset_timeout),
        output_dir=out_dir,
        euler_step_size=float(args.euler_step_size),
        success_hold_seconds=float(args.success_hold_seconds),
        record_video=bool(args.record_video),
        record_display=str(args.record_display),
        record_fps=int(args.record_fps),
    )


def _load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    model = DiffusionPolicy(
        joint_dim=int(ckpt["joint_dim"]),
        horizon=int(ckpt["horizon"]),
        emb_dim=int(model_cfg.get("emb_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        time_dim=int(model_cfg.get("time_dim", 64)),
        image_conv_channels=list(model_cfg.get("image_conv_channels", [16, 32, 64])),
        point_dim=int(data_cfg.get("point_dim", 3)),
        point_hidden_dim=int(model_cfg.get("point_hidden_dim", 64)),
        point_feature_dim=int(model_cfg.get("point_feature_dim", 64)),
        state_hidden_dim=int(model_cfg.get("state_hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def _action_smoothness(actions: np.ndarray) -> float:
    if len(actions) < 3:
        return 0.0
    vel = np.diff(actions, axis=0)
    jerk = np.diff(vel, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=-1)))


def _image_to_chw(msg: Image, target_h: int, target_w: int) -> np.ndarray:
    w = int(msg.width)
    h = int(msg.height)
    step = int(msg.step)
    if w <= 0 or h <= 0 or step <= 0:
        return np.zeros((3, target_h, target_w), dtype=np.float32)
    channels = step // w if w > 0 else 0
    if channels < 3:
        return np.zeros((3, target_h, target_w), dtype=np.float32)

    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if arr.size < h * step:
        return np.zeros((3, target_h, target_w), dtype=np.float32)
    arr = arr[: h * step].reshape((h, step))
    arr = arr[:, : w * channels].reshape((h, w, channels))
    enc = str(msg.encoding).lower()
    if enc in ("rgb8", "rgba8"):
        rgb = arr[:, :, :3]
    elif enc in ("bgr8", "bgra8"):
        rgb = arr[:, :, :3][:, :, ::-1]
    else:
        return np.zeros((3, target_h, target_w), dtype=np.float32)

    rgb = rgb.astype(np.float32) / 255.0
    if h != target_h or w != target_w:
        ys = np.linspace(0, h - 1, target_h).astype(np.int32)
        xs = np.linspace(0, w - 1, target_w).astype(np.int32)
        rgb = rgb[ys][:, xs]
    return np.transpose(rgb, (2, 0, 1))


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


def _sample_points(points: np.ndarray, point_count: int, point_dim: int, rng: np.random.Generator) -> np.ndarray:
    if points.ndim != 2 or points.shape[-1] < point_dim:
        return np.zeros((point_count, point_dim), dtype=np.float32)
    pts = points[:, :point_dim].astype(np.float32)
    n = len(pts)
    if n >= point_count:
        sel = rng.choice(n, point_count, replace=False)
        return pts[sel]
    out = np.zeros((point_count, point_dim), dtype=np.float32)
    if n > 0:
        out[:n] = pts
        fill = rng.choice(n, point_count - n, replace=True)
        out[n:] = pts[fill]
    return out


def _sample_action_sequence(
    model: DiffusionPolicy,
    obs: dict[str, torch.Tensor],
    joint_dim: int,
    horizon: int,
    sample_steps: int,
    device: str,
    euler_step_size: float,
) -> torch.Tensor:
    batch = obs["current_state"].shape[0]
    action = torch.zeros((batch, horizon, joint_dim), device=device, dtype=torch.float32)
    dt = float(euler_step_size)
    for step in range(sample_steps):
        t_value = min(step * dt, 0.999)
        t_cur = torch.full((batch,), t_value, device=device, dtype=torch.float32)
        with torch.no_grad():
            pred_velocity = model(obs, action, t_cur)
        action = action + dt * pred_velocity
    return action

class PolicyEvalNode(Node):
    def __init__(self, joint_order: list[str], image_h: int, image_w: int, point_count: int, point_dim: int, seed: int):
        super().__init__("excavator_policy_eval")
        self.joint_order = list(joint_order)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.point_count = int(point_count)
        self.point_dim = int(point_dim)
        self.rng = np.random.default_rng(seed)

        self.ready = False
        self.ready_recv_ns = 0
        self.stones_in_truck = 0
        self.latest_episode_meta: dict[str, Any] = {}
        self.driver_image_msg: Image | None = None
        self.bucket_image_msg: Image | None = None
        self.lidar_msg: PointCloud2 | None = None
        self.current_joint_msg: JointState | None = None

        self.driver_recv_ns = 0
        self.bucket_recv_ns = 0
        self.lidar_recv_ns = 0
        self.joint_recv_ns = 0

        self.cmd_publisher = self.create_publisher(JointState, "/excavator/cmd_joint", 50)
        self.reset_publisher = self.create_publisher(Int32, "/excavator/reset", 10)

        self.create_subscription(Bool, "/excavator/ready", self._on_ready, 10)
        self.create_subscription(Int32, "/excavator/stones_in_truck", self._on_stones_in_truck, 10)
        self.create_subscription(String, "/excavator/episode_meta", self._on_episode_meta, 10)
        self.create_subscription(Image, "/excavator/camera_driver/rgb", self._on_driver_image, 10)
        self.create_subscription(Image, "/excavator/camera_bucket/rgb", self._on_bucket_image, 10)
        self.create_subscription(PointCloud2, "/excavator/lidar/points", self._on_lidar, 10)
        self.create_subscription(JointState, "/excavator/joint_states", self._on_joint_state, 50)

    def _recv_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _on_ready(self, msg: Bool) -> None:
        self.ready = bool(msg.data)
        self.ready_recv_ns = self._recv_ns()

    def _on_stones_in_truck(self, msg: Int32) -> None:
        self.stones_in_truck = int(msg.data)

    def _on_episode_meta(self, msg: String) -> None:
        try:
            self.latest_episode_meta = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            self.latest_episode_meta = {}

    def _on_driver_image(self, msg: Image) -> None:
        self.driver_image_msg = msg
        self.driver_recv_ns = self._recv_ns()

    def _on_bucket_image(self, msg: Image) -> None:
        self.bucket_image_msg = msg
        self.bucket_recv_ns = self._recv_ns()

    def _on_lidar(self, msg: PointCloud2) -> None:
        self.lidar_msg = msg
        self.lidar_recv_ns = self._recv_ns()

    def _on_joint_state(self, msg: JointState) -> None:
        self.current_joint_msg = msg
        self.joint_recv_ns = self._recv_ns()

    def request_reset(self) -> None:
        msg = Int32()
        msg.data = 1
        self.reset_publisher.publish(msg)

    def has_fresh_observation(self, since_ns: int = 0) -> bool:
        return (
            self.driver_image_msg is not None
            and self.bucket_image_msg is not None
            and self.lidar_msg is not None
            and self.current_joint_msg is not None
            and self.driver_recv_ns > since_ns
            and self.bucket_recv_ns > since_ns
            and self.lidar_recv_ns > since_ns
            and self.joint_recv_ns > since_ns
        )

    def build_observation(self, device: str) -> dict[str, torch.Tensor]:
        assert self.driver_image_msg is not None
        assert self.bucket_image_msg is not None
        assert self.lidar_msg is not None
        assert self.current_joint_msg is not None

        driver = _image_to_chw(self.driver_image_msg, self.image_h, self.image_w)
        bucket = _image_to_chw(self.bucket_image_msg, self.image_h, self.image_w)
        points = _sample_points(_pointcloud_xyz_array(self.lidar_msg), self.point_count, self.point_dim, self.rng)
        state = _named_vector(self.current_joint_msg.name, self.current_joint_msg.position, self.joint_order)

        return {
            "camera_driver": torch.from_numpy(driver).unsqueeze(0).to(device),
            "camera_bucket": torch.from_numpy(bucket).unsqueeze(0).to(device),
            "points": torch.from_numpy(points).unsqueeze(0).to(device),
            "current_state": torch.from_numpy(state).unsqueeze(0).to(device),
        }

    def publish_command(self, cmd: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_order)
        msg.position = [float(x) for x in cmd.tolist()]
        self.cmd_publisher.publish(msg)


def _spin_until(node: PolicyEvalNode, predicate, timeout_sec: float, sleep_sec: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_sec
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
        time.sleep(sleep_sec)
    return False


def _prepare_episode(node: PolicyEvalNode, timeout_sec: float) -> None:
    node.request_reset()
    saw_not_ready = False

    def reset_cycle_done() -> bool:
        nonlocal saw_not_ready
        if not node.ready:
            saw_not_ready = True
        return saw_not_ready and node.ready and node.has_fresh_observation(node.ready_recv_ns)

    ok = _spin_until(node, reset_cycle_done, timeout_sec)
    if not ok:
        raise TimeoutError("reset timeout: simulator did not return ready with fresh observations")


def _ensure_startup(node: PolicyEvalNode, timeout_sec: float) -> None:
    ok = _spin_until(node, lambda: node.ready and node.has_fresh_observation(node.ready_recv_ns), timeout_sec)
    if not ok:
        raise TimeoutError("startup timeout: missing ready signal or sensor observations from simulator")


def _default_report_dir(cli_output_dir: Path | None) -> Path:
    if cli_output_dir is not None:
        cli_output_dir.mkdir(parents=True, exist_ok=True)
        return cli_output_dir
    paths = get_paths()
    out_dir = paths.logs / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_report(args: EvalConfig, checkpoint: Path, device: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    success_rate = float(np.mean([1.0 if ep.get("user_success", False) else 0.0 for ep in episodes])) if episodes else 0.0
    final_stones_avg = float(np.mean([ep["final_stones_in_truck"] for ep in episodes])) if episodes else 0.0
    return {
        "checkpoint": str(checkpoint),
        "device": device,
        "episodes_requested": args.episodes,
        "episodes_completed": len(episodes),
        "max_seconds": args.max_seconds,
        "control_hz": args.control_hz,
        "sample_steps": args.sample_steps,
        "euler_step_size": args.euler_step_size,
        "success_hold_seconds": args.success_hold_seconds,
        "user_success_rate": success_rate,
        "avg_final_stones_in_truck": final_stones_avg,
        "episodes": episodes,
    }


def _write_report(report_dir: Path, report: dict[str, Any]) -> Path:
    out_path = report_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _append_inference_debug(report_dir: Path, payload: dict[str, Any]) -> Path:
    out_path = report_dir / "inference_debug.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return out_path


def _start_episode_recording(report_dir: Path, episode_index: int, display: str, fps: int) -> tuple[subprocess.Popen[str] | None, Path | None]:
    if shutil.which("ffmpeg") is None:
        print("[eval] ffmpeg not found; skip video recording", flush=True)
        return None, None
    video_dir = report_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = video_dir / f"episode_{episode_index + 1:03d}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "x11grab",
        "-draw_mouse",
        "0",
        "-framerate",
        str(max(1, fps)),
        "-i",
        display,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        print(f"[eval] recording started: {out_path}", flush=True)
        return proc, out_path
    except Exception as exc:
        print(f"[eval] failed to start recording: {exc}", flush=True)
        return None, None


def _stop_episode_recording(proc: subprocess.Popen[str] | None, out_path: Path | None) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None and proc.stdin is not None:
            proc.stdin.write("q\n")
            proc.stdin.flush()
            proc.wait(timeout=10.0)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()
    if out_path is not None:
        print(f"[eval] recording saved: {out_path}", flush=True)


def _sequence_start_from_previous(full_seq: np.ndarray, last_cmd: np.ndarray | None) -> tuple[np.ndarray, int]:
    if full_seq.ndim != 2 or len(full_seq) == 0:
        raise ValueError("full_seq must be a non-empty 2D array")
    if last_cmd is None:
        return full_seq, 0
    diff = full_seq - last_cmd.reshape(1, -1)
    sq_dist = np.sum(np.square(diff), axis=1)
    start_idx = int(np.argmin(sq_dist))
    return full_seq[start_idx:], start_idx


def _display_checkpoint_label(checkpoint: Path) -> str:
    parts = checkpoint.parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return checkpoint.name


def _draw_status(
    screen,
    font,
    small_font,
    episode_idx: int,
    total_episodes: int,
    ready: bool,
    model_enabled: bool,
    control_hz: float,
    stones: int,
    checkpoint_label: str,
) -> None:
    screen.fill((20, 22, 26))
    title = font.render("Excavator Policy Eval", True, (230, 230, 230))
    screen.blit(title, (24, 20))

    state_color = (50, 200, 80) if ready else (220, 70, 70)
    state_text = "sim: ready" if ready else "sim: not ready"
    infer_color = (50, 200, 80) if model_enabled and ready else (230, 180, 70) if model_enabled else (180, 180, 180)
    infer_text = "policy: ON" if model_enabled else "policy: OFF"

    lines = [
        (state_text, state_color),
        (infer_text, infer_color),
        (f"episode: {episode_idx + 1}/{total_episodes}", (220, 220, 220)),
        (f"control_hz: {control_hz:.1f}", (220, 220, 220)),
        (f"stones_in_truck: {stones}", (220, 220, 220)),
        (f"model: {checkpoint_label}", (180, 180, 180)),
    ]
    y = 80
    for text, color in lines:
        screen.blit(small_font.render(text, True, color), (24, y))
        y += 34

    hint_y = 300
    hints = [
        "m: start / finish one episode",
        "a: start auto evaluation",
        "s / f: mark last finished episode succeed / fail",
        "r: manual reset environment",
        "q: quit evaluation",
    ]
    for idx, hint in enumerate(hints):
        screen.blit(small_font.render(hint, True, (180, 180, 180)), (24, hint_y + idx * 30))
    pygame.display.flip()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_model(args.checkpoint, device)
    checkpoint_label = _display_checkpoint_label(args.checkpoint)
    data_cfg = cfg["data"]
    joint_order = list(data_cfg["joint_order"])

    pygame.init()
    pygame.display.set_caption("Excavator Policy Eval")
    screen = pygame.display.set_mode((560, 460))
    font = pygame.font.SysFont("monospace", 28)
    small_font = pygame.font.SysFont("monospace", 22)

    rclpy.init()
    node = PolicyEvalNode(
        joint_order=joint_order,
        image_h=int(data_cfg["image_height"]),
        image_w=int(data_cfg["image_width"]),
        point_count=int(data_cfg["point_count"]),
        point_dim=int(data_cfg.get("point_dim", 3)),
        seed=int(data_cfg.get("seed", 42)),
    )

    try:
        _ensure_startup(node, args.startup_timeout)
        print(f"[eval] connected: ready={node.ready} stones={node.stones_in_truck}", flush=True)

        joint_dim = int(model.joint_dim)
        horizon = int(model.horizon)
        period = 1.0 / max(args.control_hz, 1e-6)
        report_dir = _default_report_dir(args.output_dir)
        episodes = []
        model_enabled = False
        episode_active = False
        awaiting_user_label = False
        episode_idx = 0
        ep_start = 0.0
        action_history: list[np.ndarray] = []
        control_steps = 0
        inference_count_total = 0
        inference_count_episode = 0
        last_cmd = np.zeros((joint_dim,), dtype=np.float32)
        predicted_chunk: np.ndarray | None = None
        chunk_step = 0
        pending_manual_reset = False
        pending_finished_report: dict[str, Any] | None = None
        auto_mode = False
        record_proc: subprocess.Popen[str] | None = None
        record_path: Path | None = None
        stones_nonzero_since: float | None = None

        def _build_finished_episode(*, finished_by: str, finished_video_path: Path | None) -> dict[str, Any]:
            duration = float(time.monotonic() - ep_start)
            total_stones = int(node.latest_episode_meta.get("stone_count", 0) or 0)
            success_threshold = int(math.ceil(total_stones * 0.10)) if total_stones > 0 else 0
            final_stones = int(node.stones_in_truck)
            action_np = np.stack(action_history, axis=0) if action_history else np.zeros((1, joint_dim), dtype=np.float32)
            return {
                "episode_index": episode_idx,
                "duration_sec": duration,
                "control_steps": control_steps,
                "control_hz_effective": float(control_steps / duration) if duration > 1e-6 else 0.0,
                "final_stones_in_truck": final_stones,
                "stone_count_total": total_stones,
                "success_threshold_stones": success_threshold,
                "auto_success": bool(final_stones >= success_threshold) if success_threshold > 0 else False,
                "action_smoothness": _action_smoothness(action_np),
                "last_command": last_cmd.tolist(),
                "episode_meta": node.latest_episode_meta,
                "finished_by": finished_by,
                "video_path": None if finished_video_path is None else str(finished_video_path),
            }

        print("[eval] waiting for 'm' to start episode 1", flush=True)

        while rclpy.ok():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        raise KeyboardInterrupt
                    if event.key == pygame.K_r:
                        pending_manual_reset = True
                    elif event.key == pygame.K_a:
                        if awaiting_user_label:
                            print("[eval] episode finished already; waiting for label or auto finalize", flush=True)
                        elif episode_active:
                            print("[eval] auto evaluation can only start when no episode is running", flush=True)
                        elif episode_idx >= args.episodes:
                            print(f"[eval] already collected {args.episodes} episodes; press q to quit", flush=True)
                        elif not node.ready:
                            print("[eval] cannot start auto evaluation: simulator not ready", flush=True)
                        else:
                            auto_mode = True
                            episode_active = True
                            model_enabled = True
                            ep_start = time.monotonic()
                            action_history = []
                            control_steps = 0
                            inference_count_episode = 0
                            last_cmd = np.zeros((joint_dim,), dtype=np.float32)
                            predicted_chunk = None
                            chunk_step = 0
                            stones_nonzero_since = None
                            if args.record_video:
                                record_proc, record_path = _start_episode_recording(
                                    report_dir=report_dir,
                                    episode_index=episode_idx,
                                    display=args.record_display,
                                    fps=args.record_fps,
                                )
                            print(f"[eval] auto episode {episode_idx + 1} started", flush=True)
                    elif event.key == pygame.K_m:
                        if awaiting_user_label:
                            print("[eval] episode finished already; press 's' or 'f' first", flush=True)
                        elif not episode_active:
                            if episode_idx >= args.episodes:
                                print(f"[eval] already collected {args.episodes} episodes; press q to quit", flush=True)
                            elif not node.ready:
                                print("[eval] cannot start episode: simulator not ready", flush=True)
                            else:
                                episode_active = True
                                model_enabled = True
                                ep_start = time.monotonic()
                                action_history = []
                                control_steps = 0
                                inference_count_episode = 0
                                last_cmd = np.zeros((joint_dim,), dtype=np.float32)
                                predicted_chunk = None
                                chunk_step = 0
                                stones_nonzero_since = None
                                auto_mode = False
                                if args.record_video:
                                    record_proc, record_path = _start_episode_recording(
                                        report_dir=report_dir,
                                        episode_index=episode_idx,
                                        display=args.record_display,
                                        fps=args.record_fps,
                                    )
                                print(f"[eval] episode {episode_idx + 1} started", flush=True)
                        else:
                            episode_active = False
                            model_enabled = False
                            finished_video_path = record_path
                            _stop_episode_recording(record_proc, record_path)
                            record_proc = None
                            record_path = None
                            pending_finished_report = _build_finished_episode(
                                finished_by="manual_stop",
                                finished_video_path=finished_video_path,
                            )
                            final_stones = int(node.stones_in_truck)
                            total_stones = int(node.latest_episode_meta.get("stone_count", 0) or 0)
                            duration = float(pending_finished_report["duration_sec"])
                            awaiting_user_label = True
                            print(
                                f"[eval] episode {episode_idx + 1} finished. "
                                f"press 's' for succeed or 'f' for fail. "
                                f"stones={final_stones}/{total_stones} duration={duration:.2f}s",
                                flush=True,
                            )
                    elif event.key == pygame.K_s:
                        if awaiting_user_label and pending_finished_report is not None:
                            pending_finished_report["user_result"] = "success"
                            pending_finished_report["user_success"] = True
                            episodes.append(pending_finished_report)
                            episode_idx += 1
                            awaiting_user_label = False
                            pending_finished_report = None
                            out_path = _write_report(report_dir, _build_report(args, args.checkpoint, device, episodes))
                            print(f"[eval] intermediate report saved: {out_path}", flush=True)
                            print(f"[eval] episode {episode_idx} marked succeed; resetting env", flush=True)
                            _prepare_episode(node, args.reset_timeout)
                            print(f"[eval] waiting for 'm' to start episode {episode_idx + 1}", flush=True)
                        else:
                            print("[eval] no finished episode awaiting label", flush=True)
                    elif event.key == pygame.K_f:
                        if awaiting_user_label and pending_finished_report is not None:
                            pending_finished_report["user_result"] = "fail"
                            pending_finished_report["user_success"] = False
                            episodes.append(pending_finished_report)
                            episode_idx += 1
                            awaiting_user_label = False
                            pending_finished_report = None
                            out_path = _write_report(report_dir, _build_report(args, args.checkpoint, device, episodes))
                            print(f"[eval] intermediate report saved: {out_path}", flush=True)
                            print(f"[eval] episode {episode_idx} marked fail; resetting env", flush=True)
                            _prepare_episode(node, args.reset_timeout)
                            print(f"[eval] waiting for 'm' to start episode {episode_idx + 1}", flush=True)
                        else:
                            print("[eval] no finished episode awaiting label", flush=True)

            step_start = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.0)
            _draw_status(
                screen=screen,
                font=font,
                small_font=small_font,
                episode_idx=min(episode_idx, args.episodes - 1) if args.episodes > 0 else 0,
                total_episodes=args.episodes,
                ready=node.ready,
                model_enabled=model_enabled,
                control_hz=args.control_hz,
                stones=node.stones_in_truck,
                checkpoint_label=checkpoint_label,
            )

            if pending_manual_reset:
                print("[eval] manual reset requested", flush=True)
                episode_active = False
                model_enabled = False
                auto_mode = False
                awaiting_user_label = False
                pending_finished_report = None
                predicted_chunk = None
                chunk_step = 0
                stones_nonzero_since = None
                _stop_episode_recording(record_proc, record_path)
                record_proc = None
                record_path = None
                _prepare_episode(node, args.reset_timeout)
                pending_manual_reset = False
                print(f"[eval] waiting for 'm' to start episode {episode_idx + 1}", flush=True)
                continue

            if not episode_active:
                time.sleep(min(period, 0.05))
                continue

            if int(node.stones_in_truck) > 0:
                if stones_nonzero_since is None:
                    stones_nonzero_since = step_start
            else:
                stones_nonzero_since = None

            success_hold_reached = (
                stones_nonzero_since is not None
                and (step_start - stones_nonzero_since) >= args.success_hold_seconds
            )

            if success_hold_reached:
                episode_active = False
                model_enabled = False
                finished_video_path = record_path
                _stop_episode_recording(record_proc, record_path)
                record_proc = None
                record_path = None
                final_stones = int(node.stones_in_truck)
                pending_finished_report = _build_finished_episode(
                    finished_by="stones_hold_success",
                    finished_video_path=finished_video_path,
                )
                pending_finished_report["user_result"] = "success"
                pending_finished_report["user_success"] = True
                episodes.append(pending_finished_report)
                episode_idx += 1
                out_path = _write_report(report_dir, _build_report(args, args.checkpoint, device, episodes))
                print(
                    f"[eval] episode {episode_idx} finished early by stones hold: "
                    f"stones={final_stones} result=success saved={out_path}",
                    flush=True,
                )
                pending_finished_report = None
                stones_nonzero_since = None
                if episode_idx >= args.episodes:
                    auto_mode = False
                    print(f"[eval] evaluation completed: {episode_idx}/{args.episodes}", flush=True)
                else:
                    _prepare_episode(node, args.reset_timeout)
                    if auto_mode:
                        episode_active = True
                        model_enabled = True
                        ep_start = time.monotonic()
                        action_history = []
                        control_steps = 0
                        inference_count_episode = 0
                        last_cmd = np.zeros((joint_dim,), dtype=np.float32)
                        predicted_chunk = None
                        chunk_step = 0
                        stones_nonzero_since = None
                        if args.record_video:
                            record_proc, record_path = _start_episode_recording(
                                report_dir=report_dir,
                                episode_index=episode_idx,
                                display=args.record_display,
                                fps=args.record_fps,
                            )
                        print(f"[eval] auto episode {episode_idx + 1} started", flush=True)
                    else:
                        print(f"[eval] waiting for 'm' to start episode {episode_idx + 1}", flush=True)
                continue

            if (step_start - ep_start) >= args.max_seconds:
                episode_active = False
                model_enabled = False
                finished_video_path = record_path
                _stop_episode_recording(record_proc, record_path)
                record_proc = None
                record_path = None
                final_stones = int(node.stones_in_truck)
                pending_finished_report = _build_finished_episode(
                    finished_by="timeout",
                    finished_video_path=finished_video_path,
                )
                if auto_mode:
                    pending_finished_report["user_result"] = "success" if final_stones > 0 else "fail"
                    pending_finished_report["user_success"] = bool(final_stones > 0)
                    episodes.append(pending_finished_report)
                    episode_idx += 1
                    out_path = _write_report(report_dir, _build_report(args, args.checkpoint, device, episodes))
                    print(
                        f"[eval] auto episode {episode_idx} finished: "
                        f"stones={final_stones} result={pending_finished_report['user_result']} saved={out_path}",
                        flush=True,
                    )
                    pending_finished_report = None
                    stones_nonzero_since = None
                    if episode_idx >= args.episodes:
                        auto_mode = False
                        print(f"[eval] auto evaluation completed: {episode_idx}/{args.episodes}", flush=True)
                    else:
                        _prepare_episode(node, args.reset_timeout)
                        episode_active = True
                        model_enabled = True
                        ep_start = time.monotonic()
                        action_history = []
                        control_steps = 0
                        inference_count_episode = 0
                        last_cmd = np.zeros((joint_dim,), dtype=np.float32)
                        predicted_chunk = None
                        chunk_step = 0
                        stones_nonzero_since = None
                        if args.record_video:
                            record_proc, record_path = _start_episode_recording(
                                report_dir=report_dir,
                                episode_index=episode_idx,
                                display=args.record_display,
                                fps=args.record_fps,
                            )
                        print(f"[eval] auto episode {episode_idx + 1} started", flush=True)
                else:
                    awaiting_user_label = True
                    print(f"[eval] episode {episode_idx + 1} reached max_seconds. press 's' or 'f' to label it", flush=True)
                continue

            if not node.ready or not node.has_fresh_observation():
                time.sleep(min(period, 0.05))
                continue

            if predicted_chunk is None or chunk_step >= len(predicted_chunk):
                obs = node.build_observation(device)
                sampled_seq = _sample_action_sequence(
                    model=model,
                    obs=obs,
                    joint_dim=joint_dim,
                    horizon=horizon,
                    sample_steps=args.sample_steps,
                    device=device,
                    euler_step_size=args.euler_step_size,
                )
                full_seq = sampled_seq[0].detach().cpu().numpy().astype(np.float32)
                inference_count_total += 1
                inference_count_episode += 1
                predicted_chunk, matched_start_idx = _sequence_start_from_previous(
                    full_seq=full_seq,
                    last_cmd=None if control_steps == 0 else last_cmd,
                )
                if len(predicted_chunk) == 0:
                    predicted_chunk = full_seq[-1:].copy()
                    matched_start_idx = max(len(full_seq) - 1, 0)
                if inference_count_total % 10 == 0:
                    current_state_np = obs["current_state"][0].detach().cpu().numpy().astype(np.float32)
                    debug_payload = {
                        "episode_index": episode_idx,
                        "inference_count_total": inference_count_total,
                        "inference_count_episode": inference_count_episode,
                        "control_steps_before_inference": control_steps,
                        "stones_in_truck": int(node.stones_in_truck),
                        "ready": bool(node.ready),
                        "euler_step_size": float(args.euler_step_size),
                        "sample_steps": int(args.sample_steps),
                        "current_state": current_state_np.tolist(),
                        "predicted_action_seq": full_seq.tolist(),
                        "matched_start_idx": int(matched_start_idx),
                        "executed_action_seq": predicted_chunk.tolist(),
                    }
                    debug_path = _append_inference_debug(report_dir, debug_payload)
                    print(
                        f"[eval] debug snapshot saved: {debug_path} "
                        f"(episode={episode_idx + 1}, inference_total={inference_count_total}, start_idx={matched_start_idx})",
                        flush=True,
                    )
                chunk_step = 0

            cmd = predicted_chunk[chunk_step]
            chunk_step += 1
            node.publish_command(cmd)
            last_cmd = cmd
            action_history.append(cmd.copy())
            control_steps += 1

            elapsed = time.monotonic() - step_start
            if elapsed < period:
                time.sleep(period - elapsed)

        report = _build_report(args, args.checkpoint, device, episodes)
        out_path = _write_report(report_dir, report)
        print(f"[eval] report saved: {out_path}", flush=True)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        print("[eval] interrupted by user, saving current report before exit", flush=True)
        if "record_proc" in locals():
            _stop_episode_recording(record_proc, record_path)
        if "report_dir" in locals():
            report = _build_report(args, args.checkpoint, device, episodes if "episodes" in locals() else [])
            out_path = _write_report(report_dir, report)
            print(f"[eval] report saved: {out_path}", flush=True)
            print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    finally:
        if "record_proc" in locals():
            _stop_episode_recording(record_proc, record_path)
        pygame.quit()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
