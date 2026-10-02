import numpy as np
import pytest

import torch

from oap_supcon.data import PoseData, normalize_pose


def test_identity_leakage_is_rejected():
    x = np.zeros((3, 4, 5, 2), np.float32)
    visibility = np.ones((3, 4, 5), np.float32)
    data = PoseData(x, visibility, np.array([1, 1, 2]), np.array(["train", "probe", "gallery"]), np.array(["a", "b", "c"]), np.array(["n"] * 3), np.array(["v"] * 3))
    with pytest.raises(ValueError, match="identity leakage"):
        data.validate()


def test_normalization_centers_coco_pose_on_pelvis():
    x = torch.zeros(1, 3, 17, 2)
    x[..., 0] = torch.arange(17)
    x[:, :, 11] = torch.tensor([10.0, 4.0])
    x[:, :, 12] = torch.tensor([14.0, 4.0])
    visibility = torch.ones(1, 3, 17)
    normalized = normalize_pose(x, visibility)
    pelvis_midpoint = (normalized[:, :, 11] + normalized[:, :, 12]) / 2
    assert torch.allclose(pelvis_midpoint, torch.zeros_like(pelvis_midpoint), atol=1e-6)


def _padded_dataset(count: int = 4, padded: int = 40, real: int = 25) -> PoseData:
    x = np.zeros((count, padded, 17, 2), np.float32)
    visibility = np.zeros((count, padded, 17), np.float32)
    rng = np.random.default_rng(0)
    frame_counts = np.full(count, real, np.int32)
    for i in range(count):
        x[i, :real] = rng.normal(size=(real, 17, 2)).astype(np.float32)
        visibility[i, :real] = 1.0
    return PoseData(
        x, visibility, np.array([1, 1, 2, 2][:count]),
        np.array(["train", "train", "gallery", "probe"][:count]),
        np.array([str(i) for i in range(count)]), np.array(["n"] * count),
        np.array(["v"] * count), frame_counts,
    )


def test_clip_sampling_returns_dense_windows_with_no_padding():
    """The padded tail must never reach the trunk when a clip length is set."""
    from oap_supcon.data import PoseDataset

    data = _padded_dataset()
    dataset = PoseDataset(data, np.arange(len(data.labels)), clip_length=16, clip_mode="random", seed=3)
    for item in range(len(dataset)):
        x, visibility, length, _, _ = dataset[item]
        assert x.shape[0] == 16 and length == 16
        assert torch.all(visibility.sum(dim=-1) > 0), "a sampled clip contains a padding frame"


def test_clip_sampling_wraps_sequences_shorter_than_the_window():
    from oap_supcon.data import PoseDataset

    data = _padded_dataset(count=1, padded=40, real=7)
    x, visibility, length, _, _ = PoseDataset(data, np.array([0]), clip_length=16, seed=0)[0]
    assert x.shape[0] == 16 and length == 16
    assert torch.all(visibility.sum(dim=-1) > 0)


def test_confidence_can_be_kept_out_of_the_pose_geometry():
    """A soft visibility channel must not shrink joints toward the pelvis."""
    x = torch.zeros(1, 2, 17, 2)
    x[:, :, 9] = torch.tensor([6.0, 8.0])
    x[:, :, 11] = torch.tensor([-1.0, 0.0])
    x[:, :, 12] = torch.tensor([1.0, 0.0])
    visibility = torch.full((1, 2, 17), 0.5)
    visibility[:, :, 11:13] = 1.0

    weighted = normalize_pose(x, visibility, weight_coordinates=True)
    unweighted = normalize_pose(x, visibility, weight_coordinates=False)
    # Same direction either way; only the radius differs.
    wrist_w, wrist_u = weighted[0, 0, 9], unweighted[0, 0, 9]
    assert torch.allclose(wrist_w / wrist_w.norm(), wrist_u / wrist_u.norm(), atol=1e-5)
    hip_ratio_w = weighted[0, 0, 9].norm() / weighted[0, 0, 12].norm()
    hip_ratio_u = unweighted[0, 0, 9].norm() / unweighted[0, 0, 12].norm()
    assert hip_ratio_w < hip_ratio_u * 0.75, "confidence no longer distorts relative limb length"


def test_absent_joints_stay_at_the_origin_without_coordinate_weighting():
    x = torch.ones(1, 2, 17, 2)
    visibility = torch.ones(1, 2, 17)
    visibility[:, :, 3] = 0.0
    unweighted = normalize_pose(x, visibility, weight_coordinates=False)
    assert torch.allclose(unweighted[:, :, 3], torch.zeros(1, 2, 2), atol=1e-6)
