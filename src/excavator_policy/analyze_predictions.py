from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from excavator_policy.config import load_config
from excavator_policy.dataset import build_dataset_from_config
from excavator_policy.model import DiffusionPolicy
from excavator_policy.model_small import SmallDiffusionPolicy
from excavator_sim.common import get_paths


@dataclass
class AnalyzeConfig:
    checkpoint: Path
    split_file: Path | None
    split_name: str
    batch_size: int
    max_samples: int
    sample_steps: int
    euler_step_size: float
    hesitation_threshold: float
    output_dir: Path | None


def parse_args() -> AnalyzeConfig:
    parser = argparse.ArgumentParser(description="Analyze predicted future command sequences against validation data.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-file", default="")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all samples from the selected split")
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--euler-step-size", type=float, default=0.1)
    parser.add_argument("--hesitation-threshold", type=float, default=0.01, help="Mean transition-norm threshold to flag a sequence as low-motion / hesitant")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    return AnalyzeConfig(
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        split_file=Path(args.split_file).expanduser().resolve() if args.split_file else None,
        split_name=str(args.split),
        batch_size=int(args.batch_size),
        max_samples=int(args.max_samples),
        sample_steps=int(args.sample_steps),
        euler_step_size=float(args.euler_step_size),
        hesitation_threshold=float(args.hesitation_threshold),
        output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
    )


def _load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    if "image_conv_channels" in model_cfg or "emb_dim" in model_cfg:
        model = SmallDiffusionPolicy(
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
    else:
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


def _resolve_split_file(run_dir: Path, split_file: Path | None) -> Path:
    if split_file is not None:
        return split_file
    return run_dir / "split.json"


def _load_allowed_runs(split_file: Path, split_name: str) -> list[str] | None:
    if split_name == "all":
        return None
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    key = "train_runs" if split_name == "train" else "val_runs"
    return list(payload.get(key, []))


def _sample_action_sequence(
    model: torch.nn.Module,
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


def _collate_items(items: list[tuple[dict[str, torch.Tensor], torch.Tensor]]):
    obs_list, action_list = zip(*items)
    obs = {
        "camera_driver": torch.stack([x["camera_driver"] for x in obs_list], dim=0),
        "camera_bucket": torch.stack([x["camera_bucket"] for x in obs_list], dim=0),
        "points": torch.stack([x["points"] for x in obs_list], dim=0),
        "current_state": torch.stack([x["current_state"] for x in obs_list], dim=0),
    }
    action = torch.stack(action_list, dim=0)
    return obs, action


def _mean_transition_norm(seq: np.ndarray) -> float:
    if seq.shape[0] < 2:
        return 0.0
    delta = np.diff(seq, axis=0)
    return float(np.mean(np.linalg.norm(delta, axis=-1)))


def _default_output_dir(cli_dir: Path | None) -> Path:
    if cli_dir is not None:
        cli_dir.mkdir(parents=True, exist_ok=True)
        return cli_dir
    paths = get_paths()
    out_dir = paths.logs / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = _load_model(args.checkpoint, device)
    data_cfg = cfg["data"]
    split_file = _resolve_split_file(args.checkpoint.parent, args.split_file)
    allowed_runs = _load_allowed_runs(split_file, args.split_name)
    dataset = build_dataset_from_config(data_cfg, allowed_runs=allowed_runs)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found for split={args.split_name} with split file {split_file}")

    max_samples = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    indices = list(range(max_samples))
    joint_dim = len(data_cfg["joint_order"])
    horizon = int(data_cfg["horizon"])

    per_step_abs_err = np.zeros((horizon, joint_dim), dtype=np.float64)
    per_step_sq_err = np.zeros((horizon, joint_dim), dtype=np.float64)
    num_samples = 0

    true_delta_abs_sum = np.zeros((horizon - 1, joint_dim), dtype=np.float64)
    pred_delta_abs_sum = np.zeros((horizon - 1, joint_dim), dtype=np.float64)
    num_delta_samples = 0

    true_hesitation = 0
    pred_hesitation = 0
    run_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "hesitation_count": 0})

    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        batch_items = [dataset[i] for i in batch_indices]
        obs, target = _collate_items(batch_items)
        obs = {k: v.to(device) for k, v in obs.items()}
        target = target.to(device)
        pred = _sample_action_sequence(
            model=model,
            obs=obs,
            joint_dim=joint_dim,
            horizon=horizon,
            sample_steps=args.sample_steps,
            device=device,
            euler_step_size=args.euler_step_size,
        )

        target_np = target.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        abs_err = np.abs(pred_np - target_np)
        sq_err = np.square(pred_np - target_np)
        per_step_abs_err += abs_err.sum(axis=0)
        per_step_sq_err += sq_err.sum(axis=0)
        num_samples += pred_np.shape[0]

        true_delta = np.abs(np.diff(target_np, axis=1))
        pred_delta = np.abs(np.diff(pred_np, axis=1))
        true_delta_abs_sum += true_delta.sum(axis=0)
        pred_delta_abs_sum += pred_delta.sum(axis=0)
        num_delta_samples += pred_np.shape[0]

        for local_i, batch_idx in enumerate(batch_indices):
            run_dir, _ = dataset.index[batch_idx]
            run_name = run_dir.name
            target_seq = target_np[local_i]
            pred_seq = pred_np[local_i]
            true_motion = _mean_transition_norm(target_seq)
            pred_motion = _mean_transition_norm(pred_seq)
            if true_motion < args.hesitation_threshold:
                true_hesitation += 1
                run_stats[run_name]["hesitation_count"] += 1
            if pred_motion < args.hesitation_threshold:
                pred_hesitation += 1
            run_stats[run_name]["count"] += 1

    per_step_mae = (per_step_abs_err / max(num_samples, 1)).tolist()
    per_step_rmse = np.sqrt(per_step_sq_err / max(num_samples, 1)).tolist()
    overall_mae = float(per_step_abs_err.sum() / max(num_samples * horizon * joint_dim, 1))
    overall_rmse = float(np.sqrt(per_step_sq_err.sum() / max(num_samples * horizon * joint_dim, 1)))
    true_delta_abs_mean = (true_delta_abs_sum / max(num_delta_samples, 1)).tolist()
    pred_delta_abs_mean = (pred_delta_abs_sum / max(num_delta_samples, 1)).tolist()

    run_hesitation = []
    for run_name, stats in sorted(run_stats.items()):
        ratio = float(stats["hesitation_count"] / max(stats["count"], 1))
        run_hesitation.append(
            {
                "run": run_name,
                "sample_count": int(stats["count"]),
                "hesitation_count": int(stats["hesitation_count"]),
                "hesitation_ratio": ratio,
            }
        )
    run_hesitation.sort(key=lambda x: x["hesitation_ratio"], reverse=True)

    report = {
        "checkpoint": str(args.checkpoint),
        "device": device,
        "split_file": str(split_file),
        "split_name": args.split_name,
        "num_samples": int(num_samples),
        "sample_steps": int(args.sample_steps),
        "euler_step_size": float(args.euler_step_size),
        "hesitation_threshold": float(args.hesitation_threshold),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "per_step_mae": per_step_mae,
        "per_step_rmse": per_step_rmse,
        "true_delta_abs_mean": true_delta_abs_mean,
        "pred_delta_abs_mean": pred_delta_abs_mean,
        "true_hesitation_ratio": float(true_hesitation / max(num_samples, 1)),
        "pred_hesitation_ratio": float(pred_hesitation / max(num_samples, 1)),
        "top_hesitation_runs": run_hesitation[:20],
    }

    out_dir = _default_output_dir(args.output_dir)
    out_path = out_dir / "prediction_analysis.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[analyze] checkpoint={args.checkpoint}")
    print(f"[analyze] split={args.split_name} samples={num_samples}")
    print(f"[analyze] overall_mae={overall_mae:.6f} overall_rmse={overall_rmse:.6f}")
    print(f"[analyze] true_hesitation_ratio={report['true_hesitation_ratio']:.4f}")
    print(f"[analyze] pred_hesitation_ratio={report['pred_hesitation_ratio']:.4f}")
    if run_hesitation:
        print("[analyze] top hesitation runs:")
        for row in run_hesitation[:10]:
            print(
                f"  {row['run']}: ratio={row['hesitation_ratio']:.3f} "
                f"({row['hesitation_count']}/{row['sample_count']})"
            )
    print(f"[analyze] report saved: {out_path}")


if __name__ == "__main__":
    main()
