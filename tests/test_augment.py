import torch

from oap_supcon.augment import anatomical_parts, corrupt, temporal_crop_view


def test_paper_partition_has_exactly_five_anatomical_regions():
    assert len(anatomical_parts(17)) == 5
    assert len(anatomical_parts(18)) == 5
    assert len(anatomical_parts(25)) == 5


def test_random_joint_mask_updates_coordinates_and_visibility():
    x = torch.ones(2, 10, 17, 2)
    visibility = torch.ones(2, 10, 17)
    generator = torch.Generator().manual_seed(11)
    corrupted, mask = corrupt(x, visibility, "random_joint", 0.3, generator)
    assert torch.all(corrupted[mask == 0] == 0)
    assert torch.all(corrupted[mask == 1] == 1)
    assert 0.25 <= float(1 - mask.mean()) <= 0.35


def test_random_joint_mask_is_independent_across_time():
    x = torch.ones(1, 20, 17, 2)
    visibility = torch.ones(1, 20, 17)
    _, mask = corrupt(x, visibility, "random_joint", 0.5, torch.Generator().manual_seed(11))
    assert torch.any(mask[0].amin(dim=0) != mask[0].amax(dim=0))


def test_temporal_mask_is_contiguous():
    x = torch.ones(1, 20, 17, 2)
    visibility = torch.ones(1, 20, 17)
    corrupted, mask = corrupt(x, visibility, "temporal", 0.5, torch.Generator().manual_seed(2))
    missing_frames = torch.nonzero(mask[0].sum(dim=1) == 0, as_tuple=False).flatten()
    assert len(missing_frames) == 10
    assert torch.all(torch.diff(missing_frames) == 1)
    assert torch.all(corrupted[:, missing_frames] == 0)


def test_temporal_crop_views_are_independent_and_keep_fixed_shape():
    x = torch.arange(24, dtype=torch.float32).view(1, 24, 1, 1).repeat(2, 1, 5, 2)
    visibility = torch.ones(2, 24, 5)
    generator = torch.Generator().manual_seed(5)
    first, first_mask = temporal_crop_view(x, visibility, generator)
    second, second_mask = temporal_crop_view(x, visibility, generator)
    assert first.shape == second.shape == x.shape
    assert first_mask.shape == second_mask.shape == visibility.shape
    assert not torch.allclose(first, second)
