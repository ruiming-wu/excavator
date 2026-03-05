from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from excavator_policy.dataset import build_dataset
from excavator_policy.model import DiffusionPolicy, diffusion_loss


def _collate(batch):
    obs_list, act_list = zip(*batch)
    max_prop = max([x["proprio"].numel() for x in obs_list] + [a.numel() for a in act_list])

    def _pad(x, n):
        if x.numel() >= n:
            return x[:n]
        return torch.cat([x, torch.zeros(n - x.numel(), dtype=x.dtype)], dim=0)

    obs = {
        "rgb": torch.stack([x["rgb"] for x in obs_list], dim=0),
        "points": torch.stack([x["points"] for x in obs_list], dim=0),
        "proprio": torch.stack([_pad(x["proprio"], max_prop) for x in obs_list], dim=0),
    }
    action = torch.stack([_pad(a, max_prop) for a in act_list], dim=0)
    return obs, action


def parse_args():
    parser = argparse.ArgumentParser(description="Train excavation imitation policy")
    parser.add_argument("--raw-root", default=os.environ.get("EXCAVATOR_DATA_RAW_DIR", "data/raw"))
    parser.add_argument("--runs-dir", default=os.environ.get("EXCAVATOR_RUNS_DIR", "runs"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = build_dataset(args.raw_root, window=args.window, horizon=args.horizon)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found under {args.raw_root}. Record trajectories first.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=_collate)

    sample_obs, sample_action = dataset[0]
    proprio_dim = max(sample_obs["proprio"].numel(), sample_action.numel())
    action_dim = proprio_dim

    model = DiffusionPolicy(action_dim=action_dim, proprio_dim=proprio_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.runs_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for obs, action in loader:
            obs = {k: v.to(device) for k, v in obs.items()}
            action = action.to(device)
            loss = diffusion_loss(model, obs, action)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item())
        mean_loss = total / max(len(loader), 1)
        print(f"epoch={epoch+1}/{args.epochs} loss={mean_loss:.6f}")

    ckpt = out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
            "args": vars(args),
        },
        ckpt,
    )
    (out_dir / "metrics.json").write_text(json.dumps({"final_loss": mean_loss}, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
