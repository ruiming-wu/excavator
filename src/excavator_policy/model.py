from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:  # pragma: no cover - handled at runtime when model is instantiated
    ResNet18_Weights = None
    resnet18 = None


class ResNet18ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, pretrained: bool = False):
        super().__init__()
        if resnet18 is None:
            raise ImportError("torchvision is required for the ResNet-18 image encoder. Please install torchvision.")
        weights = None
        if pretrained:
            if ResNet18_Weights is None:
                raise ImportError("Installed torchvision does not expose ResNet18_Weights. Please update torchvision.")
            weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.proj(feat)


class PointNetEncoder(nn.Module):
    def __init__(self, point_dim: int = 3, hidden_dims: list[int] | None = None, out_dim: int = 512):
        super().__init__()
        hidden_dims = hidden_dims or [64, 128, 256, 512]
        layers = []
        prev = point_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
            ])
            prev = h
        self.point_mlp = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Linear(prev, out_dim),
            nn.ReLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        point_feat = self.point_mlp(points)
        global_feat = point_feat.max(dim=1).values
        return self.proj(global_feat)


class JointStateEncoder(nn.Module):
    def __init__(self, joint_dim: int, hidden_dims: list[int] | None = None, out_dim: int = 512):
        super().__init__()
        hidden_dims = hidden_dims or [256, 512]
        dims = [joint_dim, *hidden_dims]
        layers = []
        for in_dim, out_hidden in zip(dims[:-1], dims[1:]):
            layers.extend([
                nn.Linear(in_dim, out_hidden),
                nn.ReLU(),
            ])
        if hidden_dims[-1] != out_dim:
            layers.extend([
                nn.Linear(hidden_dims[-1], out_dim),
                nn.ReLU(),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        joint_dim: int,
        image_out_dim: int = 512,
        image_pretrained: bool = False,
        point_dim: int = 3,
        point_hidden_dims: list[int] | None = None,
        point_out_dim: int = 512,
        state_hidden_dims: list[int] | None = None,
        state_out_dim: int = 512,
    ):
        super().__init__()
        self.image_encoder = ResNet18ImageEncoder(out_dim=image_out_dim, pretrained=image_pretrained)
        self.point_encoder = PointNetEncoder(point_dim=point_dim, hidden_dims=point_hidden_dims, out_dim=point_out_dim)
        self.state_encoder = JointStateEncoder(joint_dim=joint_dim, hidden_dims=state_hidden_dims, out_dim=state_out_dim)
        self.condition_dim = image_out_dim * 2 + point_out_dim + state_out_dim

    def forward(self, obs: dict) -> torch.Tensor:
        driver_f = self.image_encoder(obs["camera_driver"])
        bucket_f = self.image_encoder(obs["camera_bucket"])
        point_f = self.point_encoder(obs["points"])
        state_f = self.state_encoder(obs["current_state"])
        return torch.cat([driver_f, bucket_f, point_f, state_f], dim=-1)


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        joint_dim: int,
        horizon: int,
        hidden_dim: int = 2048,
        time_dim: int = 128,
        image_out_dim: int = 512,
        image_pretrained: bool = False,
        point_dim: int = 3,
        point_hidden_dims: list[int] | None = None,
        point_out_dim: int = 512,
        state_hidden_dims: list[int] | None = None,
        state_out_dim: int = 512,
    ):
        super().__init__()
        self.joint_dim = joint_dim
        self.horizon = horizon
        self.action_dim = joint_dim * horizon
        self.encoder = MultiModalEncoder(
            joint_dim=joint_dim,
            image_out_dim=image_out_dim,
            image_pretrained=image_pretrained,
            point_dim=point_dim,
            point_hidden_dims=point_hidden_dims,
            point_out_dim=point_out_dim,
            state_hidden_dims=state_hidden_dims,
            state_out_dim=state_out_dim,
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


def diffusion_loss(model: DiffusionPolicy, obs: dict, action_seq: torch.Tensor) -> torch.Tensor:
    batch = action_seq.shape[0]
    t = torch.rand(batch, device=action_seq.device)
    noise = torch.randn_like(action_seq)
    alpha = t.view(batch, 1, 1)
    noisy_action = (1.0 - alpha) * action_seq + alpha * noise
    pred_noise = model(obs, noisy_action, t)
    return nn.functional.mse_loss(pred_noise, noise)
