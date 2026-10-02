from __future__ import annotations

import torch
import torch.nn.functional as F


def _log_prob(features, temperature, candidates):
    if temperature <= 0:
        raise ValueError('contrastive temperature must be positive')
    features = F.normalize(features.float(), dim=-1)
    logits = features @ features.T / temperature
    safe_candidates = candidates.clone()
    safe_candidates[~safe_candidates.any(1), 0] = True
    denominator = logits.masked_fill(~safe_candidates, -torch.inf).logsumexp(1, keepdim=True)
    return logits - denominator


def supervised_contrastive(features, labels, temperature=0.1, weights=None):
    candidates = ~torch.eye(len(labels), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None]) & candidates
    valid = positives.any(1)
    log_prob = _log_prob(features, temperature, candidates)
    per_anchor = -log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1).clamp_min(1)
    weights = valid.float() if weights is None else weights.float() * valid
    return (per_anchor * weights).sum() / weights.sum().clamp_min(1e-6)


def part_contrastive(parts, reliability, labels, temperature, visibility_gating=True,
                     min_reliability=0.0):
    """Observed parts only, with normalized pair weights.

    Invisible parts are excluded from positives AND negatives. Normalizing by
    positive weight prevents the masking curriculum from shrinking the objective
    independently of recognition. The ungated control still excludes absent parts.
    """
    same = labels[:, None].eq(labels[None])
    not_self = ~torch.eye(len(labels), dtype=torch.bool, device=parts.device)
    losses = []
    for p in range(parts.shape[1]):
        confidence = reliability[:, p].detach().float()
        present = confidence > min_reliability
        candidates = not_self & present[:, None] & present[None]
        positives = same & candidates
        pair_weight = confidence[:, None] * confidence[None] if visibility_gating else torch.ones_like(candidates, dtype=torch.float32)
        weighted = positives * pair_weight
        total = weighted.sum(1)
        valid = total > 0
        if valid.any():
            log_prob = _log_prob(parts[:, p], temperature, candidates)
            per_anchor = -(log_prob * weighted).sum(1) / total.clamp_min(1e-6)
            anchor_weight = confidence[valid] if visibility_gating else torch.ones_like(confidence[valid])
            losses.append((per_anchor[valid] * anchor_weight).sum() / anchor_weight.sum().clamp_min(1e-6))
    return torch.stack(losses).mean() if losses else parts.sum() * 0


def batch_hard_triplet(features, labels, margin=0.2, sample_ids=None):
    """Train retrieval embeddings using cross-sequence hard positives.

    When sample_ids are supplied, another augmentation of the same sequence is
    not a positive. Anchors without positives or negatives contribute zero.
    """
    features = F.normalize(features.float(), dim=-1)
    distances = (2 - 2 * features @ features.T).clamp_min(0)
    positive = labels[:, None].eq(labels[None])
    positive &= ~torch.eye(len(labels), dtype=torch.bool, device=features.device)
    if sample_ids is not None:
        positive &= sample_ids[:, None].ne(sample_ids[None])
    negative = labels[:, None].ne(labels[None])
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return features.sum() * 0
    hardest_positive = distances.masked_fill(~positive, -torch.inf).max(1).values[valid]
    hardest_negative = distances.masked_fill(~negative, torch.inf).min(1).values[valid]
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def part_batch_hard_triplet(parts, reliability, labels, margin=0.2, sample_ids=None,
                            min_reliability=0.0):
    """Train part heads with the same metric objective as the global embedding.

    This is a control for whether a part-aware retrieval gain comes merely from
    training the part heads. Only observed parts participate, and another view
    of the same source sequence is excluded as a positive when ``sample_ids``
    are available.
    """
    losses = []
    not_self = ~torch.eye(len(labels), dtype=torch.bool, device=parts.device)
    for part in range(parts.shape[1]):
        present = reliability[:, part].detach() > min_reliability
        if present.sum() < 2:
            continue
        selected_labels = labels[present]
        selected_ids = sample_ids[present] if sample_ids is not None else None
        positive = selected_labels[:, None].eq(selected_labels[None])
        positive &= not_self[present][:, present]
        if selected_ids is not None:
            positive &= selected_ids[:, None].ne(selected_ids[None])
        negative = selected_labels[:, None].ne(selected_labels[None])
        if not (positive.any(1) & negative.any(1)).any():
            continue
        losses.append(batch_hard_triplet(
            parts[present, part], selected_labels, margin, selected_ids
        ))
    return torch.stack(losses).mean() if losses else parts.sum() * 0


def temporal_contrastive(first, second, temperature=0.1, labels=None):
    if temperature <= 0:
        raise ValueError('contrastive temperature must be positive')
    first, second = F.normalize(first.float(), dim=-1), F.normalize(second.float(), dim=-1)
    logits = first @ second.T / temperature
    targets = torch.arange(len(first), device=first.device)
    if labels is not None:
        diagonal = torch.eye(len(first), dtype=torch.bool, device=first.device)
        logits = logits.masked_fill(labels[:, None].eq(labels[None]) & ~diagonal, -torch.inf)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
