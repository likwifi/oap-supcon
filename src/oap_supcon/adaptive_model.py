"""Opt-in coordinate-gait candidate; not a reproduction of published GPGait++.

Each proposed change can be disabled for controlled development. Local graph
branches feed the global identity embedding, so metric_baseline trains them
without adding the part losses that hurt the previous development recipe.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .graph import spatial_adjacency
from .model import MultiStreamEncoder, _MultiScaleGraphBlock, masked_mean


def masked_softmax(logits, allowed, dim=-1):
    """An empty neighbourhood returns zeros, including under mixed precision."""
    weights = logits.float().masked_fill(~allowed, torch.finfo(torch.float32).min).softmax(dim)
    weights = weights * allowed
    return weights / weights.sum(dim, keepdim=True).clamp_min(1e-6)


class AdaptiveGraphBlock(_MultiScaleGraphBlock):
    """Anatomical edges plus a confidence-conditioned graph for every frame.

    The static partitions and temporal branches retain V2's structure. Both
    graph paths exclude absent sources; confidence supplies a source prior,
    while the target mask prevents absent joints becoming observed evidence.
    """
    def __init__(self, in_channels, out_channels, adjacency, dropout):
        super().__init__(in_channels, out_channels, adjacency, dropout)
        attention_dim = max(8, out_channels // 8)
        self.query = nn.Conv2d(in_channels, attention_dim, 1, bias=False)
        self.key = nn.Conv2d(in_channels, attention_dim, 1, bias=False)
        self.value = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.dynamic_scale = nn.Parameter(torch.tensor(0.1))
        self.attention_dim = attention_dim

    def attention_weights(self, x, confidence):
        # Compute softmax in float32 even when temporal convolutions use AMP.
        query = self.query(x).permute(0, 2, 3, 1).float()
        key = self.key(x).permute(0, 2, 1, 3).float()
        logits = torch.matmul(query, key) / math.sqrt(self.attention_dim)
        logits = logits + confidence.float().clamp_min(1e-6).log().unsqueeze(-2)
        allowed = (confidence > 0).unsqueeze(-2)
        weights = masked_softmax(logits, allowed)
        return weights * (confidence > 0).unsqueeze(-1)

    def forward(self, x, present, confidence=None):
        if confidence is None:
            confidence = present[:, 0]
        x = x * present
        adjacency = self.adjacency + 0.1 * self.edge_delta.tanh()
        weighted = adjacency[None, None] * confidence[:, :, None, None, :]
        weighted = weighted / weighted.abs().sum(-1, keepdim=True).clamp_min(1e-6)
        y = self.spatial(x)
        b, channels, t, j = y.shape
        y = torch.einsum("bkctj,btkij->bcti",
                         y.reshape(b, len(adjacency), channels // len(adjacency), t, j),
                         weighted.to(y.dtype))
        attention = self.attention_weights(x, confidence).to(x.dtype)
        dynamic = torch.einsum("bctj,btij->bcti", self.value(x), attention)
        y = y + self.dynamic_scale.tanh() * dynamic
        y = F.relu(self.spatial_norm(y)) * present
        y = self.fuse(torch.cat([branch(y) for branch in self.temporal], dim=1))
        return F.relu(y + self.residual(x)) * present


class PartAdaptiveEncoder(MultiStreamEncoder):
    """Angles, later stream fusion, adaptive graphs and anatomical refinement.

    With use_angles=False, stream_blocks=0, adaptive_graph=False, part_blocks=0
    and pooling='mean', construction and output match the V2 multistream model.
    Features from the local refinements are fused into the common feature map
    before global/part pooling, and thus receive global CE/triplet gradients.
    No extra local metric objective or pose-estimator input is required.
    """
    def __init__(self, dimensions, joints, identities, hidden_dim=64, embedding_dim=128,
                 dropout=0.1, blocks=6, cosine_scale=16.0, *, use_angles=True,
                 stream_blocks=1, adaptive_graph=True, part_blocks=1, pooling="mean_max"):
        if dimensions not in (2, 3):
            raise ValueError("part_adaptive expects 2D or 3D coordinates")
        if joints not in (17, 18, 25):
            raise ValueError("part_adaptive requires COCO-17, OpenPose-18 or Kinect-25 anatomy")
        if (not isinstance(stream_blocks, int) or stream_blocks < 0
                or not isinstance(part_blocks, int) or part_blocks < 0):
            raise ValueError("stream_blocks and part_blocks must be nonnegative integers")
        if pooling not in ("mean", "mean_max"):
            raise ValueError("pooling must be mean or mean_max")
        super().__init__(dimensions, joints, identities, hidden_dim, embedding_dim,
                         dropout, blocks, cosine_scale)
        self.use_angles = bool(use_angles)
        self.stream_blocks_count = stream_blocks
        self.adaptive_graph = bool(adaptive_graph)
        self.pooling = pooling
        adjacency = spatial_adjacency(joints)
        block_type = AdaptiveGraphBlock if adaptive_graph else _MultiScaleGraphBlock

        if self.use_angles:
            self.stems.append(nn.Sequential(nn.Conv2d(2, hidden_dim, 1, bias=False),
                                            nn.BatchNorm2d(hidden_dim), nn.ReLU()))
            self.fusion = nn.Conv2d(4 * hidden_dim, hidden_dim, 1, bias=False)

        self.stream_trunks = nn.ModuleList([
            nn.ModuleList([block_type(hidden_dim, hidden_dim, adjacency, dropout)
                           for _ in range(stream_blocks)]) for _ in self.stems
        ])
        if adaptive_graph:
            widths = [hidden_dim] + [block.spatial_norm.num_features for block in self.blocks]
            self.blocks = nn.ModuleList([
                block_type(widths[i], widths[i + 1], adjacency, dropout) for i in range(blocks)
            ])

        self.part_trunks = nn.ModuleList()
        if part_blocks:
            feature_dim = self.embedding.in_features
            regions = torch.zeros(len(self.parts), joints)
            for number, indices in enumerate(self.parts):
                regions[number, indices] = 1
                # Restrict both static and dynamic graph sources to this region.
                local_adjacency = adjacency * regions[number, :, None] * regions[number, None, :]
                self.part_trunks.append(nn.ModuleList([
                    block_type(feature_dim, feature_dim, local_adjacency, dropout)
                    for _ in range(part_blocks)
                ]))
            self.register_buffer("regions", regions)
            self.part_fusion = nn.Conv2d(2 * feature_dim, feature_dim, 1, bias=False)

    def angle_stream(self, x, visibility):
        """Cosine of the joint-parent-grandparent angle plus its reliability.

        Cosines avoid the derivative singularities of arccos. Degenerate bones
        and missing endpoints contribute neither an angle nor its confidence.
        This descriptor is rotation/translation/positive-scale invariant in 2D
        and 3D; input axes are not assumed.
        """
        x = x * (visibility > 0).unsqueeze(-1)
        parent, grandparent = self.parents, self.parents[self.parents]
        first = x - x[:, :, parent]
        second = x[:, :, grandparent] - x[:, :, parent]
        first_length, second_length = first.norm(dim=-1), second.norm(dim=-1)
        valid = (first_length > 1e-6) & (second_length > 1e-6)
        confidence = torch.minimum(visibility, visibility[:, :, parent])
        confidence = torch.minimum(confidence, visibility[:, :, grandparent]) * valid
        cosine = F.cosine_similarity(first, second, dim=-1, eps=1e-6).clamp(-1, 1)
        cosine = cosine * (confidence > 0)
        return torch.stack([cosine, confidence], dim=-1)

    def input_streams(self, x, visibility):
        streams = super().input_streams(x, visibility)
        return (*streams, self.angle_stream(x, visibility)) if self.use_angles else streams

    def _block(self, block, x, visibility):
        present = (visibility > 0).unsqueeze(1).to(x.dtype)
        if isinstance(block, AdaptiveGraphBlock):
            return block(x, present, visibility)
        return block(x, present)

    def encode_joints(self, x, visibility):
        present = (visibility > 0).unsqueeze(1).to(x.dtype)
        streams = []
        for stem, trunk, stream in zip(self.stems, self.stream_trunks, self.input_streams(x, visibility)):
            y = stem(stream.permute(0, 3, 1, 2)) * present
            for block in trunk:
                y = self._block(block, y, visibility)
            streams.append(y)
        y = self.fusion(torch.cat(streams, dim=1)) * present
        for block in self.blocks:
            y = self._block(block, y, visibility)
        if self.part_trunks:
            local = torch.zeros_like(y)
            for region, trunk in zip(self.regions, self.part_trunks):
                confidence = visibility * region
                partial = y * (confidence > 0).unsqueeze(1)
                for block in trunk:
                    partial = self._block(block, partial, confidence)
                local = local + partial
            local = local / self.regions.sum(0).clamp_min(1)[None, None, None, :]
            y = F.relu(y + self.part_fusion(torch.cat([y, local], dim=1))) * present
        return y.permute(0, 2, 3, 1)

    def pool_features(self, features, visibility):
        mean = masked_mean(features, visibility, (1, 2))
        if self.pooling == "mean":
            return mean
        present = (visibility > 0).unsqueeze(-1)
        maximum = features.masked_fill(~present, torch.finfo(features.dtype).min).amax(dim=(1, 2))
        maximum = torch.where(present.any(dim=(1, 2)), maximum, torch.zeros_like(maximum))
        return 0.5 * (mean + maximum)

    def forward(self, x, visibility, frame_mask=None):
        if frame_mask is not None:
            visibility = visibility * frame_mask.unsqueeze(-1)
        return super().forward(x, visibility, frame_mask)
