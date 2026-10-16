import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from oap_supcon.adaptive_model import AdaptiveGraphBlock, masked_softmax
from oap_supcon.experiment import load_configuration, run_experiment
from oap_supcon.graph import spatial_adjacency
from oap_supcon.losses import batch_hard_triplet
from oap_supcon.model import build_encoder, encoder_from_config
from oap_supcon.smoke import make_smoke_dataset

ROOT = Path(__file__).resolve().parents[1]
CONTROL = dict(use_angles=False, stream_blocks=0, adaptive_graph=False,
               part_blocks=0, pooling="mean")


def candidate(dimensions=2, joints=17, **options):
    return build_encoder("part_adaptive", dimensions, joints, 3, 8, 16, 0,
                         blocks=2, **options)


def test_disabled_candidate_preserves_v2_initialization_outputs_and_checkpoint():
    torch.manual_seed(99)
    legacy = build_encoder("multistream", 2, 17, 3, 8, 16, 0, blocks=2).eval()
    torch.manual_seed(99)
    control = candidate(**CONTROL).eval()
    assert legacy.state_dict().keys() == control.state_dict().keys()
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, control.state_dict()[key]), key
    control.load_state_dict(legacy.state_dict(), strict=True)
    x, v = torch.randn(2, 13, 17, 2), torch.rand(2, 13, 17)
    v[:, 4:7, 5] = 0
    with torch.no_grad():
        first, second = legacy(x, v), control(x, v)
    for key in first:
        assert torch.equal(first[key], second[key]), key


@pytest.mark.parametrize("dimensions,joints", [(2, 17), (2, 18), (3, 17), (3, 25)])
def test_missing_coordinate_poison_and_padding_do_not_change_candidate(dimensions, joints):
    model = candidate(dimensions, joints).eval()
    x, v = torch.randn(2, 14, joints, dimensions), torch.rand(2, 14, joints)
    v[:, 3:6, :5] = 0
    frame_mask = torch.ones(2, 14); frame_mask[:, 10:] = 0
    poisoned = x.clone(); poisoned[v == 0] = 1e6; poisoned[:, 10:] = -1e6
    changed_confidence = v.clone(); changed_confidence[:, 10:] = 1
    with torch.no_grad():
        before = model(x, v, frame_mask)
        after = model(poisoned, changed_confidence, frame_mask)
    for key in before:
        assert torch.allclose(before[key], after[key], atol=1e-6), key
    assert (after["joint_features"][:, 10:] == 0).all()


@pytest.mark.parametrize("dimensions", [2, 3])
def test_angles_preserve_geometry_invariance_and_exclude_missing_bones(dimensions):
    model = candidate(dimensions)
    x, v = torch.randn(2, 10, 17, dimensions), torch.full((2, 10, 17), .8)
    v[:, 2:5, 5] = 0
    rotation, _ = torch.linalg.qr(torch.randn(dimensions, dimensions))
    before = model.angle_stream(x, v)
    after = model.angle_stream((x @ rotation) * 3.2 + 100, v)
    assert torch.allclose(before, after, atol=3e-5)
    altered = x.clone(); altered[v == 0] = 1e9
    assert torch.equal(before, model.angle_stream(altered, v))
    assert (before[..., 0][before[..., 1] == 0] == 0).all()
    zero = model.angle_stream(torch.zeros_like(x), v)
    assert (zero == 0).all()


def test_angle_backward_at_degenerate_and_missing_joints_is_finite():
    model = candidate(3)
    x = torch.zeros(2, 9, 17, 3, requires_grad=True)
    v = torch.ones(2, 9, 17); v[0] = 0
    model.angle_stream(x, v).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad == 0).all()


def test_frame_graph_excludes_missing_sources_and_uses_confidence_prior():
    block = AdaptiveGraphBlock(8, 8, spatial_adjacency(17), 0)
    with torch.no_grad():
        block.query.weight.zero_(); block.key.weight.zero_()
    x, v = torch.randn(2, 3, 17, 8).permute(0, 3, 1, 2), torch.ones(2, 3, 17)
    v[:, :, 0] = .1; v[:, :, 1] = 0; v[1, 2] = 0
    weights = block.attention_weights(x, v)
    assert (weights[..., 1] == 0).all()
    assert (weights[:, :, 1] == 0).all()
    assert (weights[1, 2] == 0).all()
    assert (weights[0, 0, 3, 0] / weights[0, 0, 3, 2]).item() == pytest.approx(.1)
    assert weights[0, 0, 3].sum().item() == pytest.approx(1)


def test_frame_graph_changes_with_observed_motion():
    torch.manual_seed(3)
    block = AdaptiveGraphBlock(8, 16, spatial_adjacency(17), 0)
    x, v = torch.randn(2, 8, 3, 17), torch.ones(2, 3, 17)
    a = block.attention_weights(x, v)
    x[:, :, 1, 7] += 10
    b = block.attention_weights(x, v)
    assert torch.equal(a[:, 0], b[:, 0])
    assert not torch.allclose(a[:, 1], b[:, 1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_empty_attention_is_finite_at_supported_precisions(dtype):
    logits = torch.randn(2, 3, 4, 4).to(dtype)
    weights = masked_softmax(logits, torch.zeros(2, 3, 1, 4, dtype=torch.bool))
    assert torch.isfinite(weights).all() and (weights == 0).all()


def test_masked_max_pool_excludes_absent_evidence_and_handles_empty_parts():
    model = candidate()
    features = torch.tensor([[[[1., 2.]], [[1e8, 1e8]], [[3., 4.]]]])
    v = torch.tensor([[[1.], [0.], [1.]]])
    assert torch.equal(model.pool_features(features, v), torch.tensor([[2.5, 3.5]]))
    assert (model.pool_features(features, torch.zeros_like(v)) == 0).all()


def test_global_objective_trains_every_local_branch_without_part_loss():
    torch.manual_seed(2)
    model = candidate()
    x, v = torch.randn(6, 12, 17, 2), torch.ones(6, 12, 17)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    out = model(x, v)
    loss = F.cross_entropy(out["logits"], labels) + batch_hard_triplet(out["embedding"], labels)
    loss.backward()
    assert torch.isfinite(loss)
    for trunk in model.part_trunks:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trunk.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_candidate_mixed_precision_backward_with_empty_clip_is_finite():
    model = candidate()
    x, v = torch.randn(4, 12, 17, 2), torch.ones(4, 12, 17)
    v[0] = 0; v[1, 3:8] = 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, v)
        loss = F.cross_entropy(out["logits"], torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_absent_limb_and_fully_absent_clip_emit_finite_zero_local_descriptors():
    model = candidate().eval()
    x, v = torch.randn(2, 12, 17, 2), torch.ones(2, 12, 17)
    v[0, :, [5, 7, 9]] = 0; v[1] = 0
    out = model(x, v)
    for key in out:
        assert torch.isfinite(out[key]).all(), key
    for key in ["parts", "raw_parts", "part_features"]:
        assert (out[key][0, 1] == 0).all()
        assert (out[key][1] == 0).all()


def test_recipes_isolate_features_and_remain_validation_only():
    variants = {"angles": "use_angles", "late_fusion": "stream_blocks", "adaptive": "adaptive_graph",
                "parts": "part_blocks", "pooling": "pooling"}
    control = load_configuration(ROOT, "casia_b_pose", "metric_baseline", "benchmark_v3_control")
    for variant, option in variants.items():
        cfg = load_configuration(ROOT, "casia_b_pose", "metric_baseline", f"benchmark_v3_{variant}")
        changes = {key for key in cfg["model"]["options"]
                   if cfg["model"]["options"][key] != control["model"]["options"][key]}
        assert changes == {option}
        assert cfg["method"] == control["method"]
        assert cfg["evaluation"] == control["evaluation"]
        assert cfg["validation"]["enabled"] and cfg["evaluation"]["split"] == "validation"
        assert cfg["results_root"] == "results_v3_dev"


def test_candidate_training_checkpoint_reload_and_evaluation_use_no_test_samples(tmp_path, monkeypatch):
    import oap_supcon.experiment as experiment
    from oap_supcon.data import PoseData
    monkeypatch.delenv("OAP_DATA_ROOT", raising=False)
    make_smoke_dataset(tmp_path / "data/smoke/processed/dataset.npz")
    cfg = load_configuration(ROOT, "smoke", "metric_baseline", "benchmark_v3")
    cfg["model"].update(hidden_dim=8, embedding_dim=16, dropout=0)
    cfg["model"]["options"]["blocks"] = 2
    cfg["train"].update(num_workers=0, clip_length=12, identities_per_batch=2, samples_per_identity=2, amp=False)
    cfg["validation"].update(holdout_identities=2, every_epochs=1, corruption_families=["random_joint"])
    cfg["evaluation"].update(batch_size=8, severities=[0, .3], corruption_families=["random_joint"], gallery_protocols=["clean"])
    original = experiment._embeddings
    calls = []
    def guarded(model, data, indices, *args, **kwargs):
        assert set(data.split[indices]).issubset({"val_gallery", "val_probe"})
        calls.append(len(indices))
        return original(model, data, indices, *args, **kwargs)
    monkeypatch.setattr(experiment, "_embeddings", guarded)
    run, metrics = run_experiment(tmp_path, cfg, 11, 1, "cpu", 1)
    assert calls and metrics["status"] == "complete"
    assert metrics["evaluation_split"] == "validation"
    assert {row["descriptor"] for row in metrics["results"]} == {"global", "raw_reliable", "raw_uniform"}
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    data = PoseData.load(tmp_path / "data/smoke/processed/dataset.npz")
    restored = encoder_from_config(checkpoint["config"], 2, 17, len(checkpoint["config"]["train_identities"]))
    restored.load_state_dict(checkpoint["model"], strict=True)
    outcomes = np.load(run / "probe_outcomes.npz")
    assert len(outcomes["probe_indices"]) > 0


def test_array_covers_every_architecture_seed_pair_once():
    spec = importlib.util.spec_from_file_location("architecture_array", ROOT / "scripts/run_architecture_array.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tasks = module.architecture_tasks([11, 22, 33])
    assert len(tasks) == len(set(tasks)) == 21
    assert tasks[0] == ("control", 11) and tasks[-1] == ("full", 33)
    with pytest.raises(ValueError, match="distinct"):
        module.architecture_tasks([11, 11])
