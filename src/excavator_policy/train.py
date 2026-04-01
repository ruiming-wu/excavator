from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from excavator_policy.config import load_config
from excavator_policy.dataset import build_dataset_from_config
from excavator_policy.model import DiffusionPolicy, flow_matching_loss


def _collate(batch):
    obs_list, act_list = zip(*batch)
    obs = {
        "camera_driver": torch.stack([x["camera_driver"] for x in obs_list], dim=0),
        "camera_bucket": torch.stack([x["camera_bucket"] for x in obs_list], dim=0),
        "points": torch.stack([x["points"] for x in obs_list], dim=0),
        "current_state": torch.stack([x["current_state"] for x in obs_list], dim=0),
    }
    action = torch.stack(act_list, dim=0)
    return obs, action


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train excavation diffusion policy")
    parser.add_argument("--config", default="")
    return parser.parse_args()


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _init_wandb(train_cfg: dict, full_cfg: dict, out_dir: Path, run_name: str):
    wandb_cfg = train_cfg.get("wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", True)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required by training config. Please install wandb in the current environment.") from exc

    run = wandb.init(
        project=str(wandb_cfg.get("project", "excavator_policy")),
        name=str(wandb_cfg.get("name", "")).strip() or run_name,
        mode=str(wandb_cfg.get("mode", "online")),
        dir=str(out_dir),
        config=deepcopy(full_cfg),
    )
    return run


def _list_run_names(data_cfg: dict) -> list[str]:
    aligned_root = Path(data_cfg["aligned_root"])
    run_glob = str(data_cfg.get("run_glob", "run_*"))
    runs = []
    for run_dir in sorted([p for p in aligned_root.glob(run_glob) if p.is_dir()]):
        if (run_dir / "frames.parquet").exists():
            runs.append(run_dir.name)
    return runs


def _split_runs(run_names: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not run_names:
        return [], []
    shuffled = list(run_names)
    rnd = random.Random(seed)
    rnd.shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []
    train_count = max(1, int(len(shuffled) * train_ratio))
    if train_count >= len(shuffled):
        train_count = len(shuffled) - 1
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def _count_params(module: torch.nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return {"trainable": int(trainable), "total": int(total)}


def _model_stats(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    stats = {
        "image_encoder": _count_params(model.encoder.image_encoder),
        "point_encoder": _count_params(model.encoder.point_encoder),
        "state_encoder": _count_params(model.encoder.state_encoder),
        "time_mlp": _count_params(model.time_mlp),
        "denoise": _count_params(model.denoise),
        "policy_total": _count_params(model),
    }
    if hasattr(model.encoder, "fusion"):
        stats["fusion"] = _count_params(model.encoder.fusion)
    return stats


def _build_loader(dataset, train_cfg: dict, shuffle: bool) -> DataLoader:
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", True)) and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_collate,
    )


def _save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_checkpoint(path: Path, model: torch.nn.Module, joint_dim: int, horizon: int, cfg: dict, epoch: int) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "joint_dim": joint_dim,
            "horizon": horizon,
            "config": cfg,
            "epoch": epoch,
        },
        path,
    )


def _build_scheduler(optim: torch.optim.Optimizer, train_cfg: dict):
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    if not bool(sched_cfg.get("enabled", False)):
        return None
    sched_type = str(sched_cfg.get("type", "reduce_on_plateau")).lower()
    if sched_type != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode=str(sched_cfg.get("mode", "min")),
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 15)),
        min_lr=float(sched_cfg.get("min_lr", 1e-6)),
    )


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _query_gpu_metrics(device: str) -> dict[str, float]:
    if not device.startswith("cuda"):
        return {}
    try:
        query = (
            "temperature.gpu,"
            "utilization.gpu,"
            "clocks.mem,"
            "clocks.sm"
        )
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        first_line = result.stdout.strip().splitlines()[0]
        values = [float(x.strip()) for x in first_line.split(",")]
        if len(values) != 4:
            return {}
        return {
            "system/gpu_temp_c": values[0],
            "system/gpu_util_percent": values[1],
            "system/gpu_mem_clock_mhz": values[2],
            "system/gpu_sm_clock_mhz": values[3],
        }
    except Exception:
        return {}


def main():
    args = parse_args()
    default_cfg = Path(__file__).with_name("config.yaml")
    cfg_path = args.config or str(default_cfg)
    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    seed = int(data_cfg.get("seed", 42))
    _set_seed(seed)

    device = _resolve_device(str(train_cfg.get("device", "auto")))
    configured_run_name = str(train_cfg.get("run_name", "")).strip()
    if configured_run_name:
        run_name = configured_run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = timestamp
    out_dir = Path(train_cfg.get("output_dir", "logs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _init_wandb(train_cfg, cfg, out_dir, run_name)

    all_runs = _list_run_names(data_cfg)
    if not all_runs:
        raise RuntimeError(f"No aligned runs with frames.parquet found under {data_cfg['aligned_root']}. Run align first.")

    train_ratio = float(data_cfg.get("train_ratio", 0.9))
    train_runs, val_runs = _split_runs(all_runs, train_ratio, seed)
    train_dataset = build_dataset_from_config(data_cfg, allowed_runs=train_runs, hesitation_filter_enabled=True)
    val_dataset = build_dataset_from_config(data_cfg, allowed_runs=val_runs, hesitation_filter_enabled=False) if val_runs else None
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty after run split.")

    loader = _build_loader(train_dataset, train_cfg, shuffle=True)
    val_loader = _build_loader(val_dataset, train_cfg, shuffle=False) if val_dataset is not None and len(val_dataset) > 0 else None

    joint_dim = len(data_cfg["joint_order"])
    horizon = int(data_cfg["horizon"])
    model = DiffusionPolicy(
        joint_dim=joint_dim,
        horizon=horizon,
        emb_dim=int(model_cfg.get("emb_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 512)),
        time_dim=int(model_cfg.get("time_dim", 64)),
        image_conv_channels=list(model_cfg.get("image_conv_channels", [16, 32, 64])),
        point_dim=int(data_cfg.get("point_dim", 3)),
        point_hidden_dim=int(model_cfg.get("point_hidden_dim", 64)),
        point_feature_dim=int(model_cfg.get("point_feature_dim", 64)),
        state_hidden_dim=int(model_cfg.get("state_hidden_dim", 128)),
    ).to(device)
    loss_fn = flow_matching_loss
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = _build_scheduler(optim, train_cfg)

    param_stats = _model_stats(model)
    manifest = {
        "config_path": cfg.get("_config_path", ""),
        "device": device,
        "seed": seed,
        "train_runs": train_runs,
        "val_runs": val_runs,
        "num_all_runs": len(all_runs),
        "num_train_runs": len(train_runs),
        "num_val_runs": len(val_runs),
        "num_train_samples": len(train_dataset),
        "num_val_samples": 0 if val_dataset is None else len(val_dataset),
        "train_dataset_filter": getattr(train_dataset, "filter_summary", {}),
        "val_dataset_filter": {} if val_dataset is None else getattr(val_dataset, "filter_summary", {}),
        "parameter_counts": param_stats,
        "scheduler": train_cfg.get("scheduler", {}),
    }

    _save_json(out_dir / "config.json", cfg)
    _save_json(out_dir / "split.json", {"train_runs": train_runs, "val_runs": val_runs})
    _save_json(out_dir / "model_stats.json", param_stats)
    _save_json(out_dir / "manifest.json", manifest)

    print(f"[train] run_name={run_name}")
    print(f"[train] device={device} seed={seed}")
    print(f"[train] train_runs({len(train_runs)}): {train_runs}")
    print(f"[train] val_runs({len(val_runs)}): {val_runs}")
    print(f"[train] num_train_samples={len(train_dataset)} num_val_samples={0 if val_dataset is None else len(val_dataset)}")
    train_filter = getattr(train_dataset, "filter_summary", {})
    if train_filter.get("enabled", False):
        print(
            "[train] hesitation_filter(train): "
            f"kept={train_filter.get('kept_samples', 0)}/{train_filter.get('total_candidates', 0)} "
            f"dropped={train_filter.get('dropped_samples', 0)} "
            f"full_keep_threshold={train_filter.get('full_keep_threshold', 0.02):.4f}"
        )
    for part_name, stats in param_stats.items():
        print(f"[train] params.{part_name}: trainable={stats['trainable']} total={stats['total']}")

    if wandb_run is not None:
        wandb_run.config.update(manifest, allow_val_change=True)

    epochs = int(train_cfg.get("epochs", 30))
    log_interval = int(train_cfg.get("log_interval", 10))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    save_every_epoch = bool(train_cfg.get("save_every_epoch", False))
    save_interval_epochs = int(train_cfg.get("save_interval_epochs", 25))

    best_val_loss = None
    best_val_epoch = None
    final_train_loss = 0.0
    final_val_loss = None
    history = []
    global_step = 0
    train_start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for step, (obs, action) in enumerate(loader, start=1):
            global_step += 1
            obs = {k: v.to(device, non_blocking=True) for k, v in obs.items()}
            action = action.to(device, non_blocking=True)
            loss = loss_fn(model, obs, action)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optim.step()
            total += float(loss.item())
            if step % log_interval == 0 or step == len(loader):
                elapsed = _format_elapsed(time.perf_counter() - train_start_time)
                print(
                    f"epoch={epoch + 1}/{epochs} step={step}/{len(loader)} "
                    f"train_loss={loss.item():.6f} elapsed={elapsed}"
                )
                if wandb_run is not None:
                    step_log = {
                        "train/loss_step": float(loss.item()),
                    }
                    step_log.update(_query_gpu_metrics(device))
                    wandb_run.log(step_log)
        final_train_loss = total / max(len(loader), 1)

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for obs, action in val_loader:
                    obs = {k: v.to(device, non_blocking=True) for k, v in obs.items()}
                    action = action.to(device, non_blocking=True)
                    val_total += float(loss_fn(model, obs, action).item())
            final_val_loss = val_total / max(len(val_loader), 1)
            print(f"epoch={epoch + 1}/{epochs} mean_train_loss={final_train_loss:.6f} mean_val_loss={final_val_loss:.6f}")
            if best_val_loss is None or final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_val_epoch = epoch + 1
                _save_checkpoint(out_dir / "model_best.pt", model, joint_dim, horizon, cfg, epoch + 1)
            if scheduler is not None:
                scheduler.step(final_val_loss)
        else:
            final_val_loss = None
            print(f"epoch={epoch + 1}/{epochs} mean_train_loss={final_train_loss:.6f}")
            if scheduler is not None:
                scheduler.step(final_train_loss)

        current_lr = float(optim.param_groups[0]["lr"])

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_val_epoch,
            "lr": current_lr,
        }
        history.append(epoch_metrics)
        if wandb_run is not None:
            log_payload = {
                "train/loss_epoch": final_train_loss,
                "train/epoch": epoch + 1,
                "train/lr": current_lr,
            }
            if final_val_loss is not None:
                log_payload["val/loss_epoch"] = final_val_loss
            if best_val_loss is not None:
                log_payload["val/best_loss"] = best_val_loss
            if best_val_epoch is not None:
                log_payload["val/best_epoch"] = best_val_epoch
            wandb_run.log(log_payload)

        should_save_interval = save_interval_epochs > 0 and (epoch + 1) % save_interval_epochs == 0
        if save_every_epoch or should_save_interval:
            _save_checkpoint(out_dir / f"model_epoch_{epoch + 1:03d}.pt", model, joint_dim, horizon, cfg, epoch + 1)

    _save_checkpoint(out_dir / "model_last.pt", model, joint_dim, horizon, cfg, epochs)

    metrics = {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "best_val_epoch": best_val_epoch,
        "num_train_samples": len(train_dataset),
        "num_val_samples": 0 if val_dataset is None else len(val_dataset),
        "num_train_runs": len(train_runs),
        "num_val_runs": len(val_runs),
        "train_runs": train_runs,
        "val_runs": val_runs,
        "parameter_counts": param_stats,
        "save_interval_epochs": save_interval_epochs,
        "epochs": epochs,
        "last_lr": float(optim.param_groups[0]["lr"]),
    }
    _save_json(out_dir / "metrics.json", metrics)
    _save_json(out_dir / "history.json", history)

    if wandb_run is not None:
        wandb_run.summary.update(metrics)
        wandb_run.finish()
    print(f"saved last checkpoint: {out_dir / 'model_last.pt'}")
    if best_val_epoch is not None:
        print(f"saved best checkpoint: {out_dir / 'model_best.pt'} (epoch={best_val_epoch})")


if __name__ == "__main__":
    main()
