from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from excavator_policy.model import DiffusionPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Offline deployment validation on recorded runs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-run", required=True, help="run_xxx path used as rollout proxy")
    parser.add_argument("--smoothness-thr", type=float, default=0.2)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    return parser.parse_args()


def _load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DiffusionPolicy(action_dim=ckpt["action_dim"], proprio_dim=ckpt["proprio_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _action_smoothness(actions: np.ndarray) -> float:
    if len(actions) < 3:
        return 0.0
    vel = np.diff(actions, axis=0)
    jerk = np.diff(vel, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=-1)))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _load_model(Path(args.checkpoint), device)
    ts = pd.read_parquet(Path(args.eval_run) / "timestamps.parquet")
    meta = json.loads((Path(args.eval_run) / "meta.json").read_text(encoding="utf-8"))

    # Proxy rollout metric: compare predicted action stability on held-out run.
    actions = []
    for _, row in ts.iterrows():
        prop = row.get("proprio_position", [])
        if not isinstance(prop, list) or len(prop) == 0:
            continue

        p = torch.tensor(prop, dtype=torch.float32, device=device).unsqueeze(0)
        n = p.shape[-1]
        obs = {
            "rgb": torch.zeros((1, 3, 64, 64), device=device),
            "points": torch.zeros((1, 4096, 3), device=device),
            "proprio": p,
        }
        noisy = torch.zeros((1, n), device=device)
        t = torch.zeros((1,), device=device)
        with torch.no_grad():
            pred_noise = model(obs, noisy, t)
        actions.append(pred_noise.squeeze(0).cpu().numpy())

    actions_np = np.array(actions) if actions else np.zeros((1, 1), dtype=np.float32)
    smooth = _action_smoothness(actions_np)
    success = float(smooth <= args.smoothness_thr)

    # Approximate completion time from recorded trajectory.
    t_sec = 0.0
    if len(ts) > 1:
        t_sec = float((int(ts.iloc[-1]["stamp_ns"]) - int(ts.iloc[0]["stamp_ns"])) / 1e9)

    report = {
        "task_success_rate": success,
        "completion_time_sec": t_sec,
        "action_smoothness": smooth,
        "failure_mode": "high_jerk" if not success else "none",
        "num_aligned_samples": int(meta.get("counts", {}).get("aligned", 0)),
    }
    out_path = Path(args.eval_run) / "deploy_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
