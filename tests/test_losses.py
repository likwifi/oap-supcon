import torch

from oap_supcon.losses import part_contrastive, supervised_contrastive, temporal_contrastive


def test_supervised_contrastive_is_finite_and_differentiable():
    features = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive(features, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None


def test_part_contrast_requires_reliable_anchor_and_positive():
    parts = torch.randn(3, 1, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    reliability = torch.tensor([[1.0], [0.0], [1.0]])
    loss = part_contrastive(parts, reliability, labels, temperature=0.1)
    assert float(loss) == 0.0


def test_temporal_infonce_prefers_matching_crop_pairs():
    first = torch.eye(4)
    aligned = temporal_contrastive(first, first, temperature=0.1)
    mismatched = temporal_contrastive(first, first.roll(1, dims=0), temperature=0.1)
    assert aligned < mismatched
