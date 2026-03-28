from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pygame

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from excavator_sim.common import get_paths


JOINT_ORDER = ["boom_joint", "arm_joint", "bucket_joint", "swing_joint"]


@dataclass
class ReplayConfig:
    run_dir: Path
    replay_type: str
    speed: float
    width: int
    height: int
    fps: float


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


class ReplayState:
    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.meta = json.loads((self.run_dir / "meta.json").read_text(encoding="utf-8"))

        self.driver_df = _load_table(self.run_dir, "camera_driver")
        self.bucket_df = _load_table(self.run_dir, "camera_bucket")
        self.lidar_df = _load_table(self.run_dir, "lidar")
        self.proprio_df = _load_table(self.run_dir, "proprio")
        self.action_df = _load_table(self.run_dir, "action")
        self.stones_df = _load_optional_table(self.run_dir, "stones_in_truck", ["stamp_ns", "stones_recv_ns", "stones_count"])
        self.record_df = _load_table(self.run_dir, "record_control")

        self.driver_idx = 0
        self.bucket_idx = 0
        self.lidar_idx = 0
        self.proprio_idx = 0
        self.action_idx = 0
        self.stones_idx = 0

        self.driver_image: Optional[pygame.Surface] = None
        self.bucket_image: Optional[pygame.Surface] = None
        self.lidar_points: Optional[np.ndarray] = None
        self.current_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.target_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.stones_in_truck = 0
        self.start_recv_ns = self._find_first_recv_ns()
        self.finish_recv_ns = self._find_last_recv_ns()
        self.duration_s = max(0.0, (self.finish_recv_ns - self.start_recv_ns) / 1e9) if self.finish_recv_ns else 0.0

        self.last_driver_path = ""
        self.last_bucket_path = ""
        self.last_lidar_path = ""

    def _find_first_recv_ns(self) -> int:
        values = []
        for df, col in (
            (self.driver_df, "driver_recv_ns"),
            (self.bucket_df, "bucket_recv_ns"),
            (self.lidar_df, "lidar_recv_ns"),
            (self.proprio_df, "proprio_recv_ns"),
            (self.action_df, "action_recv_ns"),
            (self.stones_df, "stones_recv_ns"),
            (self.record_df, "record_recv_ns"),
        ):
            if not df.empty and col in df.columns:
                values.append(int(df[col].min()))
        return min(values) if values else 0

    def _find_last_recv_ns(self) -> int:
        values = []
        for df, col in (
            (self.driver_df, "driver_recv_ns"),
            (self.bucket_df, "bucket_recv_ns"),
            (self.lidar_df, "lidar_recv_ns"),
            (self.proprio_df, "proprio_recv_ns"),
            (self.action_df, "action_recv_ns"),
            (self.stones_df, "stones_recv_ns"),
            (self.record_df, "record_recv_ns"),
        ):
            if not df.empty and col in df.columns:
                values.append(int(df[col].max()))
        return max(values) if values else 0

    @staticmethod
    def _as_sequence(value) -> list:
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

    @staticmethod
    def _surface_from_array(arr: np.ndarray) -> Optional[pygame.Surface]:
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
        elif arr.ndim == 2:
            rgb = np.repeat(arr[:, :, None], 3, axis=2)
        else:
            return None
        rgb_whc = np.transpose(rgb, (1, 0, 2)).copy()
        return pygame.surfarray.make_surface(rgb_whc)

    def _apply_joint_row(self, row: pd.Series, target: Dict[str, float], prefix: str) -> None:
        names = self._as_sequence(row.get(f"{prefix}_name", []))
        positions = self._as_sequence(row.get(f"{prefix}_position", []))
        if not names or not positions:
            return
        for name, pos in zip(names, positions):
            if name in target:
                target[name] = float(pos)

    def update_until(self, recv_ns: int) -> None:
        while self.driver_idx < len(self.driver_df) and int(self.driver_df.iloc[self.driver_idx]["driver_recv_ns"]) <= recv_ns:
            row = self.driver_df.iloc[self.driver_idx]
            path = self.run_dir / str(row["driver_path"])
            self.driver_image = self._surface_from_array(np.load(path))
            self.last_driver_path = path.name
            self.driver_idx += 1

        while self.bucket_idx < len(self.bucket_df) and int(self.bucket_df.iloc[self.bucket_idx]["bucket_recv_ns"]) <= recv_ns:
            row = self.bucket_df.iloc[self.bucket_idx]
            path = self.run_dir / str(row["bucket_path"])
            self.bucket_image = self._surface_from_array(np.load(path))
            self.last_bucket_path = path.name
            self.bucket_idx += 1

        while self.lidar_idx < len(self.lidar_df) and int(self.lidar_df.iloc[self.lidar_idx]["lidar_recv_ns"]) <= recv_ns:
            row = self.lidar_df.iloc[self.lidar_idx]
            path = self.run_dir / str(row["lidar_path"])
            self.lidar_points = np.load(path)
            self.last_lidar_path = path.name
            self.lidar_idx += 1

        while self.proprio_idx < len(self.proprio_df) and int(self.proprio_df.iloc[self.proprio_idx]["proprio_recv_ns"]) <= recv_ns:
            row = self.proprio_df.iloc[self.proprio_idx]
            self._apply_joint_row(row, self.current_positions, "proprio")
            self.proprio_idx += 1

        while self.action_idx < len(self.action_df) and int(self.action_df.iloc[self.action_idx]["action_recv_ns"]) <= recv_ns:
            row = self.action_df.iloc[self.action_idx]
            self._apply_joint_row(row, self.target_positions, "action")
            self.action_idx += 1

        while self.stones_idx < len(self.stones_df) and int(self.stones_df.iloc[self.stones_idx]["stones_recv_ns"]) <= recv_ns:
            row = self.stones_df.iloc[self.stones_idx]
            self.stones_in_truck = int(row.get("stones_count", 0))
            self.stones_idx += 1


class AlignedReplayState:
    def __init__(self, cfg: ReplayConfig):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.paths = get_paths()
        self.raw_run_dir = self.paths.raw_data / self.run_dir.name
        if not self.raw_run_dir.exists():
            raise FileNotFoundError(f"raw run dir not found for aligned replay: {self.raw_run_dir}")

        self.meta = json.loads((self.raw_run_dir / "meta.json").read_text(encoding="utf-8"))
        self.align_meta = {}
        align_meta_path = self.run_dir / "align_meta.json"
        if align_meta_path.exists():
            self.align_meta = json.loads(align_meta_path.read_text(encoding="utf-8"))
        self.frames_df = _load_table(self.run_dir, "frames")
        if self.frames_df.empty:
            raise ValueError(f"aligned frames table is empty: {self.run_dir}")

        self.frame_ptr = 0
        self.driver_image: Optional[pygame.Surface] = None
        self.bucket_image: Optional[pygame.Surface] = None
        self.lidar_points: Optional[np.ndarray] = None
        self.current_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.target_positions: Dict[str, float] = {j: 0.0 for j in JOINT_ORDER}
        self.stones_in_truck = 0
        self.start_recv_ns = int(self.frames_df["axis_recv_ns"].iloc[0])
        self.finish_recv_ns = int(self.frames_df["axis_recv_ns"].iloc[-1])
        self.duration_s = max(0.0, (self.finish_recv_ns - self.start_recv_ns) / 1e9)

        self.last_driver_path = ""
        self.last_bucket_path = ""
        self.last_lidar_path = ""

    @staticmethod
    def _as_sequence(value) -> list:
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

    @staticmethod
    def _surface_from_array(arr: np.ndarray) -> Optional[pygame.Surface]:
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
        elif arr.ndim == 2:
            rgb = np.repeat(arr[:, :, None], 3, axis=2)
        else:
            return None
        rgb_whc = np.transpose(rgb, (1, 0, 2)).copy()
        return pygame.surfarray.make_surface(rgb_whc)

    def _apply_joint_snapshot(self, row: pd.Series, name_key: str, pos_key: str, target: Dict[str, float]) -> None:
        names = self._as_sequence(row.get(name_key, []))
        positions = self._as_sequence(row.get(pos_key, []))
        if not names or not positions:
            return
        for name, pos in zip(names, positions):
            if name in target:
                target[name] = float(pos)

    def _load_optional_surface(self, rel_path: str) -> Optional[pygame.Surface]:
        if not rel_path:
            return None
        abs_path = self.raw_run_dir / rel_path
        if not abs_path.exists():
            return None
        return self._surface_from_array(np.load(abs_path))

    def _load_optional_points(self, rel_path: str) -> Optional[np.ndarray]:
        if not rel_path:
            return None
        abs_path = self.raw_run_dir / rel_path
        if not abs_path.exists():
            return None
        return np.load(abs_path)

    def update_until(self, recv_ns: int) -> None:
        while self.frame_ptr < len(self.frames_df) and int(self.frames_df.iloc[self.frame_ptr]["axis_recv_ns"]) <= recv_ns:
            row = self.frames_df.iloc[self.frame_ptr]

            driver_rel = str(row.get("camera_driver_path", "") or "")
            bucket_rel = str(row.get("camera_bucket_path", "") or "")
            lidar_rel = str(row.get("lidar_path", "") or "")

            driver_surface = self._load_optional_surface(driver_rel)
            if driver_surface is not None:
                self.driver_image = driver_surface
                self.last_driver_path = Path(driver_rel).name

            bucket_surface = self._load_optional_surface(bucket_rel)
            if bucket_surface is not None:
                self.bucket_image = bucket_surface
                self.last_bucket_path = Path(bucket_rel).name

            lidar_points = self._load_optional_points(lidar_rel)
            if lidar_points is not None:
                self.lidar_points = lidar_points
                self.last_lidar_path = Path(lidar_rel).name

            self._apply_joint_snapshot(row, "proprio_name", "proprio_position", self.current_positions)
            self._apply_joint_snapshot(row, "action_name", "action_position", self.target_positions)
            self.stones_in_truck = int(row.get("stones_in_truck_stones_count", self.stones_in_truck))
            self.frame_ptr += 1


def _draw_lidar(screen: pygame.Surface, rect: pygame.Rect, points: Optional[np.ndarray], font: pygame.font.Font) -> None:
    pygame.draw.rect(screen, (50, 50, 50), rect, width=2)
    screen.blit(font.render("lidar replay view", True, (180, 180, 180)), (rect.x + 12, rect.y + 8))
    if points is None or points.size == 0:
        screen.blit(font.render("lidar: no points", True, (200, 100, 100)), (rect.x + 16, rect.y + 36))
        return

    near_m = 0.1
    far_m = 5.0
    camera_pos = np.array([-2.0, 0.0, 0.0], dtype=np.float32)
    rel = points - camera_pos[None, :]
    depth = rel[:, 0]
    side = rel[:, 1]
    height = rel[:, 2]
    mask = (
        (depth > near_m)
        & (depth < far_m)
        & (np.abs(side) < 5.0)
        & (height > -3.0)
        & (height < 6.0)
    )
    depth = depth[mask]
    side = side[mask]
    height = height[mask]
    pts_used = points[mask]
    if depth.size == 0:
        return

    focal = 420.0
    cx = rect.x + rect.width * 0.5
    cy = rect.y + rect.height * 0.72
    px = cx + focal * (-side / depth)
    py = cy - focal * (height / depth)
    in_rect = (
        (px >= rect.x)
        & (px < rect.x + rect.width)
        & (py >= rect.y)
        & (py < rect.y + rect.height)
    )
    px = px[in_rect]
    py = py[in_rect]
    pts_used = pts_used[in_rect]
    if px.size == 0:
        return

    dist = np.linalg.norm(pts_used, axis=1)
    dist_norm = np.clip((dist - near_m) / (far_m - near_m), 0.0, 1.0)
    colors = np.stack(
        [
            np.clip(2.0 * dist_norm - 0.2, 0.0, 1.0),
            np.clip(1.6 - 2.0 * np.abs(dist_norm - 0.5), 0.0, 1.0),
            np.clip(1.2 - 1.5 * dist_norm, 0.0, 1.0),
        ],
        axis=1,
    )
    colors = (255.0 * colors).astype(np.uint8)
    pts_px = np.stack([px, py], axis=1).astype(np.int32)
    for p, c in zip(pts_px, colors):
        pygame.draw.circle(screen, (int(c[0]), int(c[1]), int(c[2])), p, 1)


def _resolve_run_dir(run_dir_arg: str, replay_type: str) -> Path:
    p = Path(run_dir_arg).expanduser()
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    paths = get_paths()
    base = paths.raw_data if replay_type == "raw" else paths.aligned_data
    candidate = base / run_dir_arg
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"run dir not found: {run_dir_arg}")


def parse_args() -> ReplayConfig:
    parser = argparse.ArgumentParser(description="Replay one recorded excavator run using recv time")
    parser.add_argument("--type", default="raw", choices=["raw", "aligned"], help="Replay raw run or aligned run")
    parser.add_argument("--run-dir", required=True, help="Run directory path or run_xxx name under data/raw or data/aligned")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()
    return ReplayConfig(
        run_dir=_resolve_run_dir(args.run_dir, str(args.type)),
        replay_type=str(args.type),
        speed=float(args.speed),
        width=int(args.width),
        height=int(args.height),
        fps=float(args.fps),
    )


def main() -> None:
    cfg = parse_args()
    state = ReplayState(cfg) if cfg.replay_type == "raw" else AlignedReplayState(cfg)

    pygame.init()
    screen = pygame.display.set_mode((cfg.width, cfg.height))
    pygame.display.set_caption(f"Excavator Replay ({cfg.replay_type}) - {cfg.run_dir.name}")
    font = pygame.font.SysFont("monospace", 24)
    small_font = pygame.font.SysFont("monospace", 20)
    clock = pygame.time.Clock()

    paused = False
    playback_offset_ns = 0

    try:
        while True:
            dt_s = clock.tick(cfg.fps) / 1000.0
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        quit_requested = True
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        playback_offset_ns = 0
                        state = ReplayState(cfg) if cfg.replay_type == "raw" else AlignedReplayState(cfg)
                    elif event.key == pygame.K_RIGHT:
                        playback_offset_ns = min(int(state.duration_s * 1e9), playback_offset_ns + int(1e9))
                    elif event.key == pygame.K_LEFT:
                        playback_offset_ns = max(0, playback_offset_ns - int(1e9))
                        state = ReplayState(cfg) if cfg.replay_type == "raw" else AlignedReplayState(cfg)
                        state.update_until(state.start_recv_ns + playback_offset_ns)
            if quit_requested:
                break

            if not paused:
                playback_offset_ns = min(
                    int(state.duration_s * 1e9),
                    playback_offset_ns + int(dt_s * cfg.speed * 1e9),
                )
            current_recv_ns = state.start_recv_ns + playback_offset_ns
            state.update_until(current_recv_ns)

            screen.fill((24, 24, 24))
            panel_w, panel_h = 640, 360
            driver_rect = pygame.Rect(40, 40, panel_w, panel_h)
            bucket_rect = pygame.Rect(720, 40, panel_w, panel_h)
            pygame.draw.rect(screen, (50, 50, 50), driver_rect, width=2)
            pygame.draw.rect(screen, (50, 50, 50), bucket_rect, width=2)

            if state.driver_image is not None:
                screen.blit(pygame.transform.smoothscale(state.driver_image, (panel_w, panel_h)), driver_rect.topleft)
            else:
                screen.blit(small_font.render("driver camera: no image", True, (200, 100, 100)), (driver_rect.x + 16, driver_rect.y + 16))

            if state.bucket_image is not None:
                screen.blit(pygame.transform.smoothscale(state.bucket_image, (panel_w, panel_h)), bucket_rect.topleft)
            else:
                screen.blit(small_font.render("bucket camera: no image", True, (200, 100, 100)), (bucket_rect.x + 16, bucket_rect.y + 16))

            lidar_rect = pygame.Rect(760, 430, 560, 380)
            _draw_lidar(screen, lidar_rect, state.lidar_points, small_font)

            screen.blit(font.render(f"run: {cfg.run_dir.name} ({cfg.replay_type})", True, (220, 220, 220)), (40, 430))
            screen.blit(font.render(f"stones in truck: {state.stones_in_truck}", True, (220, 220, 220)), (40, 465))
            screen.blit(font.render(f"time: {playback_offset_ns / 1e9:6.2f} / {state.duration_s:6.2f} s", True, (220, 220, 220)), (40, 500))
            screen.blit(font.render(f"speed: x{cfg.speed:.2f}  {'paused' if paused else 'playing'}", True, (220, 220, 220)), (40, 535))

            y = 580
            for j in JOINT_ORDER:
                cp = state.current_positions[j]
                tp = state.target_positions[j]
                line = f"{j:12s} cur={cp:+.3f} rad  tgt={tp:+.3f} rad"
                screen.blit(small_font.render(line, True, (220, 220, 220)), (40, y))
                y += 34

            info_y = y + 12
            screen.blit(small_font.render(f"truck center: {state.meta.get('episode_meta', {}).get('truck_bottom_center_xy', '-')}", True, (180, 180, 180)), (40, info_y))
            screen.blit(small_font.render(f"pile center: {state.meta.get('episode_meta', {}).get('pile_center_xy', '-')}", True, (180, 180, 180)), (40, info_y + 28))
            screen.blit(small_font.render(f"driver frame: {state.last_driver_path or '-'}", True, (180, 180, 180)), (40, info_y + 56))
            screen.blit(small_font.render(f"bucket frame: {state.last_bucket_path or '-'}", True, (180, 180, 180)), (40, info_y + 84))
            screen.blit(small_font.render(f"lidar frame: {state.last_lidar_path or '-'}", True, (180, 180, 180)), (40, info_y + 112))

            hint_y = info_y + 152
            screen.blit(small_font.render("space: pause/resume", True, (180, 180, 180)), (40, hint_y))
            screen.blit(small_font.render("left/right: seek -/+1s", True, (180, 180, 180)), (40, hint_y + 28))
            screen.blit(small_font.render("r: restart replay", True, (180, 180, 180)), (40, hint_y + 56))
            screen.blit(small_font.render("q: quit", True, (180, 180, 180)), (40, hint_y + 84))

            pygame.display.flip()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
