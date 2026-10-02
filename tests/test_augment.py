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


def _padded_batch(real_length: int, padded: int = 40, joints: int = 17):
    """One sequence of `real_length` frames zero-padded out to `padded`."""
    x = torch.zeros(1, padded, joints, 2)
    visibility = torch.zeros(1, padded, joints)
    x[:, :real_length] = 1.0
    visibility[:, :real_length] = 1.0
    return x, visibility, torch.tensor([real_length])


def test_temporal_severity_is_measured_against_the_real_clip_not_the_padding():
    # Regression: severity used to scale the padded buffer, so a nominal 0.5 on a
    # 20-of-40-frame clip removed anywhere between 0% and 100% of the real frames.
    x, visibility, lengths = _padded_batch(20, padded=40)
    for severity, expected in ((0.1, 2), (0.5, 10), (0.7, 14)):
        for seed in range(8):
            _, mask = corrupt(
                x, visibility, "temporal", severity, torch.Generator().manual_seed(seed), lengths
            )
            real = mask[0, :20]
            assert int((real.sum(dim=1) == 0).sum()) == expected
            assert torch.all(mask[0, 20:] == 0)  # padding untouched


def test_temporal_crop_never_samples_from_the_padding():
    # Regression: 14.8% of CASIA-B crops used to be 100% padding.
    x, visibility, lengths = _padded_batch(20, padded=40)
    for seed in range(16):
        cropped, mask = temporal_crop_view(
            x, visibility, torch.Generator().manual_seed(seed), 0.5, 0.8, lengths
        )
        assert float(mask.sum()) > 0, "crop collapsed to all-padding"
        assert torch.all(mask[0, 20:] == 0), "crop leaked into the padding region"
        assert cropped.shape == x.shape


def test_speed_perturb_keeps_the_padding_boundary():
    from oap_supcon.augment import speed_perturb

    x, visibility, lengths = _padded_batch(20, padded=40)
    perturbed, mask = speed_perturb(
        x, visibility, torch.Generator().manual_seed(3), (0.8, 1.2), lengths
    )
    assert torch.all(mask[0, 20:] == 0)
    assert torch.all(perturbed[0, 20:] == 0)
    assert float(mask[0, :20].sum()) > 0
