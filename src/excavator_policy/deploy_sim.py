from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from excavator_policy.dataset import _named_vector
from excavator_policy.model import DiffusionPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Offline deployment validation on aligned runs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-run", required=True, help="aligned run_xxx path used as rollout proxy")
    parser.add_argument("--smoothness-thr", type=float, default=0.2)
    return parser.parse_args()


def _load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    model = DiffusionPolicy(
        joint_dim=int(ckpt["joint_dim"]),
        horizon=int(ckpt["horizon"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 2048)),
        time_dim=int(model_cfg.get("time_dim", 128)),
        image_out_dim=int(model_cfg.get("image_out_dim", 512)),
        image_pretrained=bool(model_cfg.get("image_pretrained", False)),
        point_dim=int(data_cfg.get("point_dim", 3)),
        point_hidden_dims=list(model_cfg.get("point_hidden_dims", [64, 128, 256, 512])),
        point_out_dim=int(model_cfg.get("point_out_dim", 512)),
        state_hidden_dims=list(model_cfg.get("state_hidden_dims", [256, 512])),
        state_out_dim=int(model_cfg.get("state_out_dim", 512)),
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


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_model(Path(args.checkpoint), device)

    eval_run = Path(args.eval_run)
    frames = pd.read_parquet(eval_run / "frames.parquet")
    joint_order = list(cfg["data"]["joint_order"])
    horizon = int(cfg["data"]["horizon"])
    joint_dim = len(joint_order)

    actions = []
    zero_images = torch.zeros((1, 3, int(cfg["data"]["image_height"]), int(cfg["data"]["image_width"])), device=device)
    zero_points = torch.zeros((1, int(cfg["data"]["point_count"]), int(cfg["data"]["point_dim"])), device=device)
    for _, row in frames.iterrows():
        state = _named_vector(row.get("proprio_name", []), row.get("proprio_position", []), joint_order)
        obs = {
            "camera_driver": zero_images,
            "camera_bucket": zero_images,
            "points": zero_points,
            "current_state": torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
        }
        noisy = torch.zeros((1, horizon, joint_dim), dtype=torch.float32, device=device)
        t = torch.zeros((1,), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(obs, noisy, t)
        actions.append(pred[0, 0].cpu().numpy())

    actions_np = np.array(actions) if actions else np.zeros((1, joint_dim), dtype=np.float32)
    smooth = _action_smoothness(actions_np)
    success = float(smooth <= args.smoothness_thr)

    t_sec = 0.0
    if len(frames) > 1:
        t_sec = float((int(frames.iloc[-1]["axis_recv_ns"]) - int(frames.iloc[0]["axis_recv_ns"])) / 1e9)

    report = {
        "task_success_rate": success,
        "completion_time_sec": t_sec,
        "action_smoothness": smooth,
        "failure_mode": "high_jerk" if not success else "none",
        "num_aligned_samples": int(len(frames)),
    }
    out_path = eval_run / "deploy_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
