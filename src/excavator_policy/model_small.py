from __future__ import annotations

import torch
import torch.nn as nn


class SmallImageEncoder(nn.Module):
    def __init__(self, conv_channels: list[int] | None = None, out_dim: int = 64):
        super().__init__()
        conv_channels = conv_channels or [16, 32, 64]
        if len(conv_channels) != 3:
            raise ValueError(f"SmallImageEncoder expects exactly 3 conv channels, got {conv_channels}")
        c1, c2, c3 = conv_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c3, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallPointEncoder(nn.Module):
    def __init__(self, point_dim: int = 3, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        point_feat = self.net(points)
        return point_feat.mean(dim=1)


class SmallMultiModalEncoder(nn.Module):
    def __init__(
        self,
        joint_dim: int,
        emb_dim: int = 256,
        image_conv_channels: list[int] | None = None,
        point_dim: int = 3,
        point_hidden_dim: int = 64,
        point_feature_dim: int = 64,
        state_hidden_dim: int = 128,
    ):
        super().__init__()
        image_feature_dim = point_feature_dim
        self.image_encoder = SmallImageEncoder(conv_channels=image_conv_channels, out_dim=image_feature_dim)
        self.point_encoder = SmallPointEncoder(point_dim=point_dim, hidden_dim=point_hidden_dim, out_dim=point_feature_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(joint_dim, state_hidden_dim),
            nn.ReLU(),
        )
        fusion_in = image_feature_dim * 2 + point_feature_dim + state_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, emb_dim),
            nn.ReLU(),
        )
        self.condition_dim = emb_dim

    def forward(self, obs: dict) -> torch.Tensor:
        driver_f = self.image_encoder(obs["camera_driver"])
        bucket_f = self.image_encoder(obs["camera_bucket"])
        point_f = self.point_encoder(obs["points"])
        state_f = self.state_encoder(obs["current_state"])
        return self.fusion(torch.cat([driver_f, bucket_f, point_f, state_f], dim=-1))


class SmallDiffusionPolicy(nn.Module):
    def __init__(
        self,
        joint_dim: int,
        horizon: int,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        time_dim: int = 64,
        image_conv_channels: list[int] | None = None,
        point_dim: int = 3,
        point_hidden_dim: int = 64,
        point_feature_dim: int = 64,
        state_hidden_dim: int = 128,
    ):
        super().__init__()
        self.joint_dim = joint_dim
        self.horizon = horizon
        self.action_dim = joint_dim * horizon
        self.encoder = SmallMultiModalEncoder(
            joint_dim=joint_dim,
            emb_dim=emb_dim,
            image_conv_channels=image_conv_channels,
            point_dim=point_dim,
            point_hidden_dim=point_hidden_dim,
            point_feature_dim=point_feature_dim,
            state_hidden_dim=state_hidden_dim,
        )
        self.condition_dim = self.encoder.condition_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        denoise_in = self.condition_dim + self.action_dim + time_dim
        self.denoise = nn.Sequential(
            nn.Linear(denoise_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def forward(self, obs: dict, noisy_action_seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self.encoder(obs)
        t_feat = self.time_mlp(t.unsqueeze(-1))
        noisy_flat = noisy_action_seq.reshape(noisy_action_seq.shape[0], -1)
        x = torch.cat([cond, noisy_flat, t_feat], dim=-1)
        out = self.denoise(x)
        return out.view(noisy_action_seq.shape[0], self.horizon, self.joint_dim)


def diffusion_loss_small(model: SmallDiffusionPolicy, obs: dict, action_seq: torch.Tensor) -> torch.Tensor:
    batch = action_seq.shape[0]
    t = torch.rand(batch, device=action_seq.device)
    noise = torch.randn_like(action_seq)
    alpha = t.view(batch, 1, 1)
    noisy_action = (1.0 - alpha) * action_seq + alpha * noise
    pred_noise = model(obs, noisy_action, t)
    return nn.functional.mse_loss(pred_noise, noise)
