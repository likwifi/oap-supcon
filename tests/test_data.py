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


def test_invalid_frame_counts_are_rejected():
    x = np.zeros((2, 4, 5, 2), np.float32)
    visibility = np.ones((2, 4, 5), np.float32)
    data = PoseData(
        x,
        visibility,
        np.array([1, 2]),
        np.array(["train", "probe"]),
        np.array(["a", "b"]),
        np.array(["n", "n"]),
        np.array(["v", "v"]),
        np.array([4, 5]),
    )
    with pytest.raises(ValueError, match="frame_counts"):
        data.validate()
