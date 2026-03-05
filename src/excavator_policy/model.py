from __future__ import annotations

import torch
import torch.nn as nn


class MultiModalEncoder(nn.Module):
    def __init__(self, proprio_dim: int, emb_dim: int = 256):
        super().__init__()
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, emb_dim // 4),
            nn.ReLU(),
        )
        self.pc_enc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim // 4),
            nn.ReLU(),
        )
        self.prop_enc = nn.Sequential(
            nn.Linear(proprio_dim, emb_dim // 2),
            nn.ReLU(),
        )

    def forward(self, rgb: torch.Tensor, points: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        rgb_f = self.rgb_enc(rgb)
        pc_f = self.pc_enc(points).mean(dim=1)
        prop_f = self.prop_enc(proprio)
        return torch.cat([rgb_f, pc_f, prop_f], dim=-1)


class DiffusionPolicy(nn.Module):
    """A compact diffusion-style denoiser over actions conditioned on multimodal observations."""

    def __init__(self, action_dim: int, proprio_dim: int, emb_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.encoder = MultiModalEncoder(proprio_dim=proprio_dim, emb_dim=emb_dim)
        self.time_mlp = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.denoise = nn.Sequential(
            nn.Linear(emb_dim + action_dim + time_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, obs: dict, noisy_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self.encoder(obs["rgb"], obs["points"], obs["proprio"])
        t_feat = self.time_mlp(t.unsqueeze(-1))
        x = torch.cat([cond, noisy_action, t_feat], dim=-1)
        return self.denoise(x)


def diffusion_loss(model: DiffusionPolicy, obs: dict, action: torch.Tensor):
    batch = action.shape[0]
    t = torch.rand(batch, device=action.device)
    noise = torch.randn_like(action)
    alpha = t.unsqueeze(-1)
    noisy_action = (1.0 - alpha) * action + alpha * noise
    pred_noise = model(obs, noisy_action, t)
    return nn.functional.mse_loss(pred_noise, noise)
