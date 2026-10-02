import torch

from oap_supcon.graph import skeleton_edges, spatial_adjacency
from oap_supcon.model import BACKBONES, PoseEncoder, build_encoder

ANATOMICAL = ("stgcn", "transformer")


def test_every_backbone_exposes_the_same_interface():
    x, visibility = torch.randn(2, 16, 17, 2), torch.ones(2, 16, 17)
    frame_mask = torch.ones(2, 16)
    for name in BACKBONES:
        model = build_encoder(name, 2, 17, 8).eval()
        with torch.no_grad():
            out = model(x, visibility, frame_mask)
        assert out["embedding"].shape == (2, 128)
        assert out["parts"].shape == (2, 5, 128)
        assert out["reliability"].shape == (2, 5)
        assert out["logits"].shape == (2, 8)
        assert torch.isfinite(out["embedding"]).all()


def test_anatomical_backbones_are_not_permutation_equivariant():
    # The whole point: a part-level loss needs a trunk that can tell a left
    # ankle from a right wrist. The tcn control cannot, by construction.
    x, visibility = torch.randn(2, 16, 17, 2), torch.ones(2, 16, 17)
    permutation = torch.randperm(17)
    for name in ANATOMICAL:
        model = build_encoder(name, 2, 17, 8).eval()
        with torch.no_grad():
            a, b = model.embed(x, visibility), model.embed(x[:, :, permutation], visibility)
        assert not torch.allclose(a, b, atol=1e-4), f"{name} ignores joint identity"
    control = build_encoder("tcn", 2, 17, 8).eval()
    with torch.no_grad():
        a, b = control.embed(x, visibility), control.embed(x[:, :, permutation], visibility)
    assert torch.allclose(a, b, atol=1e-5)


def test_backbones_survive_fully_occluded_frames():
    # A frame with no visible joint must not produce NaN through attention.
    x, visibility = torch.randn(2, 16, 17, 2), torch.ones(2, 16, 17)
    visibility[:, 4:9] = 0.0
    visibility[1] = 0.0
    for name in BACKBONES:
        model = build_encoder(name, 2, 17, 8).eval()
        with torch.no_grad():
            out = model(x * visibility.unsqueeze(-1), visibility, torch.ones(2, 16))
        assert torch.isfinite(out["embedding"]).all(), f"{name} produced NaN"
        assert torch.isfinite(out["parts"]).all()


def test_backbones_are_differentiable():
    x, visibility = torch.randn(2, 16, 17, 2, requires_grad=True), torch.ones(2, 16, 17)
    for name in BACKBONES:
        model = build_encoder(name, 2, 17, 8)
        model(x, visibility, torch.ones(2, 16))["embedding"].sum().backward()
        assert any(p.grad is not None for p in model.parameters())


def test_skeleton_graph_is_connected_and_row_normalised():
    for joints in (17, 18, 25):
        adjacency = spatial_adjacency(joints)
        assert adjacency.shape == (3, joints, joints)
        assert float(adjacency.sum(dim=-1).max()) <= 1.0 + 1e-6
        # every joint participates in at least one subset
        assert bool((adjacency.sum(dim=(0, 2)) > 0).all())
        for a, b in skeleton_edges(joints):
            assert 0 <= a < joints and 0 <= b < joints


def test_unknown_backbone_is_rejected():
    try:
        build_encoder("resnet", 2, 17, 8)
    except ValueError as error:
        assert "unknown backbone" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_tcn_backbone_is_still_the_original_pose_encoder():
    assert BACKBONES["tcn"] is PoseEncoder
