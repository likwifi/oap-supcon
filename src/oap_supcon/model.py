from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .augment import anatomical_parts
from .graph import spatial_adjacency, skeleton_edges, center_joint


def masked_mean(features: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]):
    weights = mask.unsqueeze(-1)
    return (features * weights).sum(dim=dims) / weights.sum(dim=dims).clamp_min(1e-6)


class _GaitHeads(nn.Module):
    """Pooling, projection and classification shared by every backbone.

    Subclasses implement `encode_joints` and must call `_build_heads` *after*
    creating their own modules, so that parameter-initialisation order -- and
    therefore seeded reproducibility of the existing results -- is preserved.
    """

    def _build_heads(self, joints: int, feature_dim: int, identities: int, embedding_dim: int):
        self.joints = joints
        self.parts = anatomical_parts(joints)
        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.part_projector = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, identities)

    def encode_joints(self, x: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def embed(self, x: torch.Tensor, visibility: torch.Tensor):
        joint_features = self.encode_joints(x, visibility)
        pooled = masked_mean(joint_features, visibility, (1, 2))
        return F.normalize(self.embedding(pooled), dim=-1)

    def project(self, x: torch.Tensor, visibility: torch.Tensor):
        joint_features = self.encode_joints(x, visibility)
        pooled = masked_mean(joint_features, visibility, (1, 2))
        return F.normalize(self.projector(self.embedding(pooled)), dim=-1)

    def forward(self, x: torch.Tensor, visibility: torch.Tensor, frame_mask: torch.Tensor | None = None):
        """`frame_mask` marks real (non-padding) frames; without it the padded
        buffer length is used, which makes the part reliability weights of
        Equation 11 a proxy for clip length rather than for occlusion."""
        joint_features = self.encode_joints(x, visibility)
        pooled = masked_mean(joint_features, visibility, (1, 2))
        embedding = self.embedding(pooled)
        projection = F.normalize(self.projector(embedding), dim=-1)
        if frame_mask is None:
            real_frames = torch.full(
                (x.shape[0],), float(x.shape[1]), device=x.device, dtype=joint_features.dtype
            )
        else:
            real_frames = frame_mask.sum(dim=1).clamp_min(1.0).to(joint_features.dtype)
        part_features, raw_parts, part_projections, reliability = [], [], [], []
        for indices in self.parts:
            part_mask = visibility[:, :, indices]
            part_feature = masked_mean(joint_features[:, :, indices], part_mask, (1, 2))
            present = (part_mask.sum((1, 2)) > 0).unsqueeze(-1)
            vector = F.normalize(self.part_projector(part_feature), dim=-1)
            part_features.append(part_feature * present)
            raw_parts.append(F.normalize(part_feature, dim=-1) * present)
            part_projections.append(vector * present)
            # Equation 11 normalised over the real clip rather than the padded
            # buffer, so the weight no longer tracks sequence length. It reaches
            # 1.0 only where visibility is binary; with a continuous confidence
            # channel a fully visible part scores its mean confidence instead
            # (0.83 on CASIA-B HRNet), which scales the part loss by a roughly
            # constant factor but still drops correctly under real occlusion.
            reliability.append(
                (part_mask.sum(dim=(1, 2)) / (real_frames * len(indices))).clamp(0.0, 1.0)
            )
        return {
            "embedding": F.normalize(embedding, dim=-1),
            "projection": projection,
            "part_features": torch.stack(part_features, dim=1),
            "raw_parts": torch.stack(raw_parts, dim=1),
            "parts": torch.stack(part_projections, dim=1),
            "reliability": torch.stack(reliability, dim=1),
            "logits": self.classifier(embedding),
            "joint_features": joint_features,
        }


class PoseEncoder(_GaitHeads):
    """Small shared spatio-temporal encoder suitable for reference and smoke runs.

    Per-joint temporal convolutions with no skeleton connectivity, so it is
    permutation-equivariant over joints. Retained unchanged as the control that
    the published pre-fix results were produced with; prefer `stgcn` or
    `transformer` for any anatomical claim.
    """
    def __init__(self, dimensions: int, joints: int, identities: int, hidden_dim: int = 64, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(dimensions + 1, hidden_dim)
        self.temporal = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(),
        )
        self._build_heads(joints, hidden_dim, identities, embedding_dim)

    def encode_joints(self, x: torch.Tensor, visibility: torch.Tensor):
        inputs = torch.cat([x, visibility.unsqueeze(-1)], dim=-1)
        joint_features = self.input(inputs).permute(0, 3, 1, 2)
        joint_features = self.temporal(joint_features).permute(0, 2, 3, 1)
        return joint_features


class _STGCNBlock(nn.Module):
    """One ST-GCN unit: partitioned graph convolution then a temporal convolution."""
    def __init__(self, in_channels: int, out_channels: int, subsets: int, temporal_kernel: int = 9, dropout: float = 0.1):
        super().__init__()
        self.subsets = subsets
        self.spatial = nn.Conv2d(in_channels, out_channels * subsets, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, (temporal_kernel, 1),
                padding=((temporal_kernel - 1) // 2, 0),
            ),
            nn.BatchNorm2d(out_channels), nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor):
        residual = self.residual(x)
        y = self.spatial(x)
        batch, channels, frames, joints = y.shape
        y = y.view(batch, self.subsets, channels // self.subsets, frames, joints)
        # adjacency[k, target, source]: each target aggregates over its sources.
        y = torch.einsum("bkctj,kwj->bctw", y, adjacency)
        return F.relu(self.temporal(y) + residual)


class STGCNEncoder(_GaitHeads):
    """Spatial-temporal graph convolutional backbone (Yan et al., AAAI 2018).

    Uses the skeleton bone graph with ST-GCN spatial partitioning plus learnable
    edge importance, so joints are distinguishable by their position in the
    anatomy rather than only by their coordinates.
    """
    def __init__(self, dimensions: int, joints: int, identities: int, hidden_dim: int = 64, embedding_dim: int = 128, dropout: float = 0.1, blocks: int = 3):
        super().__init__()
        adjacency = spatial_adjacency(joints)
        self.register_buffer("adjacency", adjacency)
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones_like(adjacency)) for _ in range(blocks)]
        )
        self.input_norm = nn.BatchNorm1d((dimensions + 1) * joints)
        channels = [dimensions + 1] + [hidden_dim] * (blocks - 1) + [hidden_dim * 2]
        self.blocks = nn.ModuleList([
            _STGCNBlock(channels[i], channels[i + 1], adjacency.shape[0], dropout=dropout)
            for i in range(blocks)
        ])
        self._build_heads(joints, channels[-1], identities, embedding_dim)

    def encode_joints(self, x: torch.Tensor, visibility: torch.Tensor):
        inputs = torch.cat([x, visibility.unsqueeze(-1)], dim=-1)
        batch, frames, joints, channels = inputs.shape
        # ST-GCN normalises the raw coordinate stream per joint-channel.
        normalised = self.input_norm(
            inputs.permute(0, 2, 3, 1).reshape(batch, joints * channels, frames)
        )
        y = normalised.view(batch, joints, channels, frames).permute(0, 2, 3, 1)
        for block, importance in zip(self.blocks, self.edge_importance):
            y = block(y, self.adjacency * importance)
        return y.permute(0, 2, 3, 1)


class SkeletonTransformerEncoder(_GaitHeads):
    """Spatial attention followed by temporal convolutions over joints and frames.

    A learnable per-joint embedding breaks permutation equivariance, and
    invisible joints are excluded from spatial attention via a key padding mask.
    """
    def __init__(self, dimensions: int, joints: int, identities: int, hidden_dim: int = 64, embedding_dim: int = 128, dropout: float = 0.1, heads: int = 4, layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input = nn.Linear(dimensions + 1, hidden_dim)
        self.joint_embedding = nn.Parameter(torch.zeros(joints, hidden_dim))
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        self.spatial_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                hidden_dim, heads, hidden_dim * 4, dropout,
                batch_first=True, norm_first=True,
            ) for _ in range(layers)
        ])
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
                nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            ) for _ in range(layers)
        ])
        self._build_heads(joints, hidden_dim, identities, embedding_dim)

    def encode_joints(self, x: torch.Tensor, visibility: torch.Tensor):
        inputs = torch.cat([x, visibility.unsqueeze(-1)], dim=-1)
        batch, frames, joints, _ = inputs.shape
        y = self.input(inputs) + self.joint_embedding
        # A frame with no visible joint would make softmax collapse to NaN, so
        # keep one key open for those frames instead of masking every joint.
        padding = visibility.reshape(batch * frames, joints) <= 0
        blank = padding.all(dim=-1)
        if blank.any():
            padding = padding.clone()
            padding[blank, 0] = False
        for attention, temporal in zip(self.spatial_layers, self.temporal_convs):
            flat = y.reshape(batch * frames, joints, self.hidden_dim)
            flat = attention(flat, src_key_padding_mask=padding)
            y = flat.view(batch, frames, joints, self.hidden_dim)
            moved = y.permute(0, 2, 3, 1).reshape(batch * joints, self.hidden_dim, frames)
            moved = temporal(moved)
            y = moved.view(batch, joints, self.hidden_dim, frames).permute(0, 3, 1, 2)
        return y


class _MultiScaleGraphBlock(nn.Module):
    """Learned graph edges and three temporal receptive fields, with a residual."""
    def __init__(self, in_channels, out_channels, adjacency, dropout):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.edge_delta = nn.Parameter(torch.zeros_like(adjacency))
        self.spatial = nn.Conv2d(in_channels, out_channels * len(adjacency), 1, bias=False)
        self.spatial_norm = nn.BatchNorm2d(out_channels)
        branch = max(8, out_channels // 4)
        self.temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(),
                nn.Conv2d(branch, branch, (3, 1), padding=(dilation, 0),
                          dilation=(dilation, 1), bias=False),
            ) for dilation in (1, 2, 4)
        ])
        self.fuse = nn.Sequential(nn.Conv2d(3 * branch, out_channels, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.Dropout(dropout))
        self.residual = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x, present):
        adjacency = self.adjacency + 0.1 * self.edge_delta.tanh()
        adjacency = adjacency / adjacency.abs().sum(-1, keepdim=True).clamp_min(1.0)
        y = self.spatial(x)
        b, channels, t, j = y.shape
        y = torch.einsum("bkctj,kwj->bctw", y.reshape(b, len(adjacency), channels // len(adjacency), t, j), adjacency)
        y = F.relu(self.spatial_norm(y)) * present
        y = self.fuse(torch.cat([branch(y) for branch in self.temporal], dim=1))
        return F.relu(y + self.residual(x)) * present


class MultiStreamEncoder(_GaitHeads):
    """Joint, bone and two-lag motion streams feeding a residual graph network.

    Derived geometry exists only when BOTH endpoints are observed. Confidence
    is a separate channel. This is a candidate, not an official reproduction.
    """
    def __init__(self, dimensions, joints, identities, hidden_dim=64, embedding_dim=128,
                 dropout=0.1, blocks=6, cosine_scale=16.0):
        super().__init__()
        if blocks < 2:
            raise ValueError("multistream requires at least two graph blocks")
        parents = list(range(joints))
        root = center_joint(joints)
        visited, pending = {root}, [root]
        edges = skeleton_edges(joints)
        for parent in pending:
            neighbors = sorted({b if a == parent else a for a, b in edges if parent in (a, b)})
            for child in neighbors:
                if child not in visited:
                    parents[child] = parent
                    visited.add(child)
                    pending.append(child)
        self.register_buffer("parents", torch.tensor(parents))
        self.stems = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, hidden_dim, 1, bias=False),
                          nn.BatchNorm2d(hidden_dim), nn.ReLU())
            for channels in (dimensions + 1, dimensions + 1, 2 * dimensions + 2)
        ])
        self.fusion = nn.Conv2d(3 * hidden_dim, hidden_dim, 1, bias=False)
        widths = [hidden_dim] + [hidden_dim * (1 if i < blocks // 3 else 2 if i < 2 * blocks // 3 else 4)
                                for i in range(blocks)]
        self.blocks = nn.ModuleList([
            _MultiScaleGraphBlock(widths[i], widths[i + 1], spatial_adjacency(joints), dropout)
            for i in range(blocks)
        ])
        self._build_heads(joints, widths[-1], identities, embedding_dim)
        self.part_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(widths[-1], embedding_dim), nn.LayerNorm(embedding_dim),
                          nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)) for _ in self.parts
        ])
        del self.part_projector
        self.classifier = nn.Linear(embedding_dim, identities, bias=False)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.cosine_scale = float(cosine_scale)

    def input_streams(self, x, visibility):
        x = x * (visibility > 0).unsqueeze(-1)
        joint = torch.cat([x, visibility.unsqueeze(-1)], -1)
        bone_confidence = torch.minimum(visibility, visibility[:, :, self.parents])
        bone = (x - x[:, :, self.parents]) * (bone_confidence > 0).unsqueeze(-1)
        bone = torch.cat([bone, bone_confidence.unsqueeze(-1)], -1)
        motion = []
        for lag in (1, 2):
            velocity, confidence = torch.zeros_like(x), torch.zeros_like(visibility)
            if x.shape[1] > lag:
                confidence[:, lag:] = torch.minimum(visibility[:, lag:], visibility[:, :-lag])
                velocity[:, lag:] = (x[:, lag:] - x[:, :-lag]) / lag
            velocity = velocity * (confidence > 0).unsqueeze(-1)
            motion.extend([velocity, confidence.unsqueeze(-1)])
        return joint, bone, torch.cat(motion, -1)

    def encode_joints(self, x, visibility):
        present = (visibility > 0).unsqueeze(1).to(x.dtype)
        streams = [stem(stream.permute(0, 3, 1, 2)) * present
                   for stem, stream in zip(self.stems, self.input_streams(x, visibility))]
        y = self.fusion(torch.cat(streams, 1)) * present
        for block in self.blocks:
            y = block(y, present)
        return y.permute(0, 2, 3, 1)

    def forward(self, x, visibility, frame_mask=None):
        features = self.encode_joints(x, visibility)
        embedding = self.embedding_norm(self.embedding(masked_mean(features, visibility, (1, 2))))
        embedding = F.normalize(embedding, dim=-1)
        real = (frame_mask.sum(1) if frame_mask is not None
                else torch.full((len(x),), x.shape[1], device=x.device)).clamp_min(1)
        part_features, raw_parts, parts, reliability = [], [], [], []
        for head, indices in zip(self.part_heads, self.parts):
            confidence = visibility[:, :, indices]
            reliability.append((confidence.sum((1, 2)) / (real * len(indices))).clamp(0, 1))
            feature = masked_mean(features[:, :, indices], confidence, (1, 2))
            present = (confidence.sum((1, 2)) > 0).unsqueeze(-1)
            part_features.append(feature * present)
            raw_parts.append(F.normalize(feature, dim=-1) * present)
            vector = F.normalize(head(feature), dim=-1)
            parts.append(vector * present)
        return {"embedding": embedding, "projection": F.normalize(self.projector(embedding), dim=-1),
                "part_features": torch.stack(part_features, 1),
                "raw_parts": torch.stack(raw_parts, 1),
                "parts": torch.stack(parts, 1), "reliability": torch.stack(reliability, 1),
                "logits": self.cosine_scale * F.linear(embedding, F.normalize(self.classifier.weight, dim=-1)),
                "joint_features": features}

    def embed(self, x, visibility):
        return self(x, visibility)["embedding"]

    def project(self, x, visibility):
        return self(x, visibility)["projection"]


BACKBONES = {
    "tcn": PoseEncoder,
    "stgcn": STGCNEncoder,
    "transformer": SkeletonTransformerEncoder,
    "multistream": MultiStreamEncoder,
}


def build_encoder(backbone: str, *args, **kwargs) -> _GaitHeads:
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone}; choose from {', '.join(BACKBONES)}")
    return BACKBONES[backbone](*args, **kwargs)


def encoder_from_config(cfg: dict, dimensions: int, joints: int, identities: int):
    model = cfg["model"]
    return build_encoder(model.get("backbone", "tcn"), dimensions, joints, identities,
                         int(model["hidden_dim"]), int(model["embedding_dim"]), float(model["dropout"]),
                         **model.get("options", {}))
