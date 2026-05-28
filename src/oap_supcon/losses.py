from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1, weights: torch.Tensor | None = None):
    features = F.normalize(features, dim=-1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid = positives.any(dim=1)
    exp_logits = torch.exp(logits) * ~self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    per_anchor = -(log_prob * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    if weights is None:
        weights = torch.ones_like(per_anchor)
    weights = weights * valid
    return (per_anchor * weights).sum() / weights.sum().clamp_min(1.0)


def part_contrastive(parts: torch.Tensor, reliability: torch.Tensor, labels: torch.Tensor, temperature: float, visibility_gating: bool = True):
    """Equation 8: gate every positive pair by anchor and positive reliability."""
    parts = F.normalize(parts, dim=-1)
    losses = []
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=parts.device)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positives.sum(dim=1).clamp_min(1)
    for p in range(parts.shape[1]):
        logits = parts[:, p] @ parts[:, p].T / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * ~self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        if visibility_gating:
            pair_reliability = reliability[:, p, None] * reliability[None, :, p]
        else:
            pair_reliability = torch.ones_like(log_prob)
        weighted_positives = positives * pair_reliability
        per_anchor = -(log_prob * weighted_positives).sum(dim=1) / positive_count
        valid = weighted_positives.sum(dim=1) > 0
        if valid.any():
            losses.append(per_anchor[valid].mean())
    return torch.stack(losses).mean() if losses else parts.sum() * 0


def temporal_contrastive(first: torch.Tensor, second: torch.Tensor, temperature: float = 0.1):
    """Symmetric crop-to-crop InfoNCE corresponding to Equation 9."""
    first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    targets = torch.arange(first.shape[0], device=first.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
