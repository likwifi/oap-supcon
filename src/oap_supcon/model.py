from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .augment import anatomical_parts


def masked_mean(features: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]):
    weights = mask.unsqueeze(-1)
    return (features * weights).sum(dim=dims) / weights.sum(dim=dims).clamp_min(1.0)


class PoseEncoder(nn.Module):
    """Small shared spatio-temporal encoder suitable for reference and smoke runs."""
    def __init__(self, dimensions: int, joints: int, identities: int, hidden_dim: int = 64, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.joints = joints
        self.parts = anatomical_parts(joints)
        self.input = nn.Linear(dimensions + 1, hidden_dim)
        self.temporal = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(),
        )
        self.embedding = nn.Linear(hidden_dim, embedding_dim)
        self.projector = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        self.part_projector = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        self.classifier = nn.Linear(embedding_dim, identities)

    def encode_joints(self, x: torch.Tensor, visibility: torch.Tensor):
        inputs = torch.cat([x, visibility.unsqueeze(-1)], dim=-1)
        joint_features = self.input(inputs).permute(0, 3, 1, 2)
        joint_features = self.temporal(joint_features).permute(0, 2, 3, 1)
        return joint_features

    def embed(self, x: torch.Tensor, visibility: torch.Tensor):
        joint_features = self.encode_joints(x, visibility)
        pooled = masked_mean(joint_features, visibility, (1, 2))
        embedding = self.embedding(pooled)
        return F.normalize(embedding, dim=-1)

    def project(self, x: torch.Tensor, visibility: torch.Tensor):
        joint_features = self.encode_joints(x, visibility)
        pooled = masked_mean(joint_features, visibility, (1, 2))
        embedding = self.embedding(pooled)
        return F.normalize(self.projector(embedding), dim=-1)

    def forward(self, x: torch.Tensor, visibility: torch.Tensor):
        joint_features = self.encode_joints(x, visibility)
        pooled = masked_mean(joint_features, visibility, (1, 2))
        embedding = self.embedding(pooled)
        projection = F.normalize(self.projector(embedding), dim=-1)
        part_projections, reliability = [], []
        for indices in self.parts:
            part_mask = visibility[:, :, indices]
            part_feature = masked_mean(joint_features[:, :, indices], part_mask, (1, 2))
            part_projections.append(F.normalize(self.part_projector(part_feature), dim=-1))
            reliability.append(part_mask.mean(dim=(1, 2)))
        return {
            "embedding": F.normalize(embedding, dim=-1),
            "projection": projection,
            "parts": torch.stack(part_projections, dim=1),
            "reliability": torch.stack(reliability, dim=1),
            "logits": self.classifier(embedding),
            "joint_features": joint_features,
        }
