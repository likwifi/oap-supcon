from dataclasses import replace
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from oap_supcon.augment import corrupt, generic_view, speed_perturb, temporal_crop_view, temporal_training_views
from oap_supcon.data import PoseData, PoseDataset, IdentityBatchSampler, development_split, normalize_pose
from oap_supcon.experiment import (_descriptors, descriptor_similarity, _embeddings,
                                   load_configuration, run_experiment, validation_score)
from oap_supcon.descriptor_controls import evaluate_descriptor_controls
from oap_supcon.head_training import (
    _freeze_backbone, distillation_loss, part_variances, train_frozen_heads,
    variance_floor_loss,
)
from oap_supcon.losses import (part_batch_hard_triplet, part_contrastive,
                               supervised_contrastive, batch_hard_triplet)
from oap_supcon.model import build_encoder
from oap_supcon.provenance import configuration_fingerprint
from oap_supcon.reporting import aggregate_results
from oap_supcon.smoke import make_smoke_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("operation", ["generic", "crop", "speed", "corrupt"])
def test_noop_augmentation_preserves_fractional_confidence_geometry(operation):
    x, v = torch.ones(2, 20, 17, 2), torch.full((2, 20, 17), .8)
    gen = torch.Generator().manual_seed(11)
    if operation == "generic":
        y, mask = generic_view(x, v, gen, noise=0, scale_range=(1, 1))
    elif operation == "crop":
        y, mask = temporal_crop_view(x, v, gen, 1, 1)
    elif operation == "speed":
        y, mask = speed_perturb(x, v, gen, (1, 1))
    else:
        y, mask = corrupt(x, v, "random_joint", 1e-10, gen)
    assert torch.equal(mask, v)
    assert torch.allclose(y.abs(), x)


@pytest.mark.parametrize("family", ["random_joint", "dynamic_joint", "body_part", "temporal"])
def test_corruption_only_changes_removed_coordinates(family):
    x = torch.randn(2, 20, 17, 2)
    v = torch.full(x.shape[:3], .73)
    y, mask = corrupt(x, v, family, .5, torch.Generator().manual_seed(1))
    assert torch.equal(y[mask > 0], x[mask > 0])
    assert (y[mask == 0] == 0).all()


def test_no_part_masking_applies_to_temporal_branch(monkeypatch):
    import oap_supcon.augment as module
    observed = []
    original = module.occlusion_view
    def wrapped(*args, **kwargs):
        observed.append(kwargs.get("families"))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "occlusion_view", wrapped)
    x, v = torch.ones(2, 12, 17, 2), torch.ones(2, 12, 17)
    temporal_training_views(x, v, "complete_to_partial_no_part_mask", .5, torch.Generator())
    assert observed == [("random_joint", "temporal")]
    observed.clear()
    temporal_training_views(x, v, "two_partial", .5, torch.Generator())
    assert len(observed) == 2


def test_low_confidence_root_center_is_translation_invariant():
    x = torch.randn(2, 12, 17, 2)
    v = torch.full(x.shape[:3], .2)
    a = normalize_pose(x, v, False)
    b = normalize_pose(x + 100, v, False)
    assert torch.allclose(a, b, atol=2e-5)


def _data():
    labels, ids, split, views = [], [], [], []
    for identity in range(1, 9):
        for view in ("000", "090"):
            for seq in (1, 2, 5, 6):
                labels.append(identity)
                ids.append(f"{identity:03d}-nm-{seq:02d}-{view}")
                views.append(view)
                split.append("train" if identity <= 6 else "gallery" if seq < 5 else "probe")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(len(labels), 16, 17, 2)).astype(np.float32)
    return PoseData(x, np.full(x.shape[:3], .8, np.float32), np.array(labels), np.array(split),
                    np.array(ids), np.full(len(labels), "NM"), np.array(views), np.full(len(labels), 16))


def test_development_split_is_identity_disjoint_fixed_and_keeps_test_untouched():
    data = _data()
    cfg = {"dataset": {"name": "casia_b_pose"}, "validation": {"enabled": True, "holdout_identities": 2, "split_seed": 11}}
    dev, held = development_split(data, cfg)
    assert held == development_split(data, cfg)[1]
    assert not set(held) & set(dev.labels[dev.split == "train"])
    test = np.isin(data.split, ["gallery", "probe"])
    assert np.array_equal(dev.split[test], data.split[test])
    assert np.array_equal(data.split[data.labels <= 6], np.full(48, "train"))
    assert dev.x is data.x
    dev.validate()
    bad = replace(dev, split=dev.split.copy())
    bad.split[np.flatnonzero(bad.split == "val_probe")[0]] = "train"
    with pytest.raises(ValueError, match="validation"):
        bad.validate()


def test_long_random_clips_do_not_wrap(monkeypatch):
    data = _data()
    import oap_supcon.data as module
    monkeypatch.setattr(module, "normalize_pose", lambda x, *args: x)
    data.x[0, :, 0, 0] = np.arange(16)
    dataset = PoseDataset(data, np.array([0]), clip_length=10, clip_mode="random")
    for _ in range(100):
        x, _, _, _, _ = dataset[0]
        assert (torch.diff(x[:, 0, 0]) == 1).all()


def test_sampler_selects_distinct_views_and_distinct_sequences():
    labels = np.repeat([0, 1], 8)
    views = np.tile(["000", "000", "045", "045", "090", "090", "180", "180"], 2)
    sampler = IdentityBatchSampler(labels, 2, 4, 11, views)
    for batch in sampler:
        for identity in (0, 1):
            selected = [i for i in batch if labels[i] == identity]
            assert len(np.unique(views[selected])) == 4


def test_reliable_matching_ignores_absent_parts_and_has_global_fallback():
    # A missing part contains arbitrary large garbage; it must not affect scores.
    probe = np.array([[[1., 0., 1.], [500., -700., 0.]]], np.float32)
    gallery = np.array([[[1., 0., 1.], [0., 1., 1.]], [[0., 1., 1.], [1., 0., 1.]]], np.float32)
    assert np.allclose(descriptor_similarity(probe, gallery, "reliable_parts"), [[1, 0]])
    probe[0, 1] = [0, 1, 1]
    assert np.allclose(descriptor_similarity(probe, gallery, "reliable_parts", .5), [[1, 0]])


def test_multistream_masks_bone_and_velocity_endpoints():
    model = build_encoder("multistream", 2, 17, 4, 8, 16, 0, blocks=2)
    x, v = torch.randn(2, 10, 17, 2), torch.ones(2, 10, 17)
    v[:, 4:6, 5] = 0
    streams = model.input_streams(x, v)
    _, bone, motion = streams
    assert (bone[..., :2][bone[..., 2] == 0] == 0).all()
    assert (motion[..., :2][motion[..., 2] == 0] == 0).all()
    assert (motion[..., 3:5][motion[..., 5] == 0] == 0).all()
    # An absent endpoint cannot inject its arbitrary input coordinate into a stream.
    changed = x.clone(); changed[v == 0] = 1e6
    for a, b in zip(streams, model.input_streams(changed, v)):
        assert torch.equal(a, b)


def test_multistream_absent_part_has_no_descriptor_and_all_losses_backward():
    model = build_encoder("multistream", 2, 17, 2, 8, 16, 0, blocks=2)
    x, v = torch.randn(4, 12, 17, 2), torch.ones(4, 12, 17)
    v[:, :, [5, 7, 9]] = 0
    out = model(x, v)
    assert (out["parts"][:, 1] == 0).all()
    assert (out["raw_parts"][:, 1] == 0).all()
    assert (out["part_features"][:, 1] == 0).all()
    labels = torch.tensor([0, 0, 1, 1])
    loss = part_contrastive(out["parts"], out["reliability"], labels, .01)
    loss = loss + batch_hard_triplet(out["embedding"], labels) + supervised_contrastive(out["projection"], labels, .01)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_descriptor_controls_share_forwards_and_emit_all_p0_variants():
    data = _data()
    cfg = {
        "dataset": {"name": "casia_b_pose", "exclude_identical_view": True},
        "pose": {"weight_coordinates": False},
        "train": {"clip_length": 12},
        "validation": {"enabled": True, "holdout_identities": 2, "split_seed": 11},
        "evaluation": {
            "split": "validation", "batch_size": 8, "clip_length": None,
            "corruption_families": ["body_part"], "severities": [0.0, 0.3],
            "gallery_protocols": ["clean"], "corruption_seed": 17,
        },
    }
    data, _ = development_split(data, cfg)
    # With two graph blocks the raw part width is 4 * hidden_dim = 32.
    model = build_encoder("multistream", 2, 17, 4, 8, 32, 0, blocks=2).eval()
    rows, outcomes, probe_indices = evaluate_descriptor_controls(
        model, data, cfg, torch.device("cpu"), seed=11, realizations=1,
        random_head_seeds=list(range(1001, 1011)), fusion_weights=[0, .5],
    )
    descriptors = {row["descriptor"] for row in rows}
    assert {"global", "raw_reliable", "raw_uniform", "frozen_head_reliable"} <= descriptors
    assert len([name for name in descriptors if name.startswith("random_head_")]) == 10
    assert {row["fusion_weight"] for row in rows if row["descriptor"] == "raw_reliable"} == {0, .5}
    assert len(probe_indices) == len(data.indices(["val_probe"]))
    assert any(name.startswith("raw_reliable_w0.5_clean_gallery_body_part_0.3") for name in outcomes)


def test_p1_head_losses_freeze_the_backbone_and_penalize_collapse():
    model = build_encoder("multistream", 2, 17, 4, 8, 32, 0, blocks=2)
    trainable = _freeze_backbone(model)
    assert trainable == sum(parameter.numel() for parameter in model.part_heads.parameters())
    assert all(parameter.requires_grad for parameter in model.part_heads.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("part_heads")
    )
    x, visibility = torch.randn(8, 12, 17, 2), torch.ones(8, 12, 17)
    output = model(x, visibility)
    loss = distillation_loss(
        output["parts"], output["embedding"], output["reliability"], .05
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.part_heads.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("part_heads")
    )
    collapsed = torch.zeros(5)
    assert variance_floor_loss(collapsed, 1e-4) == pytest.approx(1.0)
    spread = part_variances(output["parts"].detach(), output["reliability"], .05)
    assert spread.shape == (5,)
    assert torch.isfinite(spread).all()


def test_p1_head_only_training_writes_selected_checkpoint(tmp_path):
    data = _data()
    cfg = {
        "dataset": {"name": "casia_b_pose", "exclude_identical_view": True},
        "pose": {"weight_coordinates": False},
        "model": {"backbone": "multistream", "hidden_dim": 8, "embedding_dim": 32,
                  "dropout": 0.0, "options": {"blocks": 2}},
        "method": {"augmentation": "mixed_partial"},
        "train": {
            "clip_length": 12, "identities_per_batch": 2, "samples_per_identity": 2,
            "diverse_views": True, "num_workers": 0, "amp": False,
            "weight_decay": 1e-4, "train_severity_start": .05,
            "train_severity": .3, "severity_ramp_fraction": .5,
            "min_part_reliability": .05, "triplet_margin": .2, "gradient_clip": 5,
        },
        "validation": {
            "enabled": True, "holdout_identities": 2, "split_seed": 11,
            "every_epochs": 1, "descriptor": "reliable_parts",
            "corruption_families": ["body_part"], "severity": .3,
            "corruption_seed": 17, "robust_weight": .25,
        },
        "evaluation": {"batch_size": 8, "clip_length": None, "part_weight": .5},
    }
    data, _ = development_split(data, cfg)
    model = build_encoder("multistream", 2, 17, 4, 8, 32, 0, blocks=2)
    history, _, selected, trainable = train_frozen_heads(
        model, data, cfg, torch.device("cpu"), 11, "triplet_variance", 1,
        tmp_path, cfg,
    )
    assert selected == 1
    assert trainable > 0
    assert len(history) == 1 and "validation" in history[0]
    assert (tmp_path / "checkpoint-best.pt").exists()
    assert (tmp_path / "checkpoint-last.pt").exists()


def test_absent_parts_are_not_contrastive_negatives():
    parts = torch.randn(4, 1, 8, requires_grad=True)
    rel = torch.tensor([[1.], [1.], [1.], [0.]])
    labels = torch.tensor([0, 0, 1, 2])
    loss = part_contrastive(parts, rel, labels, .01)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(parts.grad).all()
    assert (parts.grad[3] == 0).all()
    other = parts.detach().clone(); other[3] = 1000
    assert torch.allclose(loss, part_contrastive(other, rel, labels, .01))


def test_pair_weight_normalization_keeps_loss_scale_under_uniform_confidence():
    parts = torch.randn(4, 2, 8)
    labels = torch.tensor([0, 0, 1, 1])
    a = part_contrastive(parts, torch.ones(4, 2), labels, .1)
    b = part_contrastive(parts, torch.full((4, 2), .3), labels, .1)
    assert torch.allclose(a, b, atol=1e-5)


def test_triplet_excludes_same_sequence_augmentations():
    x = torch.randn(4, 8, requires_grad=True)
    labels, ids = torch.tensor([0, 0, 1, 1]), torch.tensor([2, 2, 3, 3])
    loss = batch_hard_triplet(x, labels, sample_ids=ids)
    assert loss == 0
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_part_triplet_trains_only_observed_part_heads():
    parts = torch.randn(8, 2, 12, requires_grad=True)
    reliability = torch.ones(8, 2)
    reliability[:, 1] = 0
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    sample_ids = torch.tensor([10, 11, 20, 21, 10, 11, 20, 21])
    loss = part_batch_hard_triplet(parts, reliability, labels, .2, sample_ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert parts.grad[:, 0].abs().sum() > 0
    assert parts.grad[:, 1].abs().sum() == 0


def test_masks_and_embeddings_do_not_depend_on_evaluation_batch_size():
    data = _data()
    model = build_encoder("multistream", 2, 17, 6, 8, 16, 0, blocks=2).eval()
    indices = np.arange(8)
    a, _, _ = _embeddings(model, data, indices, torch.device("cpu"), "random_joint", .3, 17, batch_size=2)
    b, _, _ = _embeddings(model, data, indices, torch.device("cpu"), "random_joint", .3, 17, batch_size=5)
    for key in a:
        assert np.allclose(a[key], b[key], atol=1e-5)


def test_checksum_includes_protocol_metadata_and_visibility_is_finite():
    data = _data()
    changed = replace(data, views=np.full(len(data.labels), "180"))
    assert data.checksum() != changed.checksum()
    data.visibility[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        data.validate()


def test_fingerprint_excludes_training_seed_but_includes_representation():
    cfg = load_configuration(ROOT, "casia_b_pose", "oap_v2", "benchmark_v2")
    assert configuration_fingerprint({**cfg, "seed": 11}) == configuration_fingerprint({**cfg, "seed": 22})
    altered = copy.deepcopy(cfg); altered["model"]["embedding_dim"] = 128
    assert configuration_fingerprint(cfg) != configuration_fingerprint(altered)


def test_factorial_recipe_inherits_benchmark_and_defines_loss_matrix():
    expected = {
        "metric_baseline": (0.0, 0.0),
        "metric_global_supcon": (0.2, 0.0),
        "metric_part_supcon": (0.0, 0.2),
        "oap_v2": (0.2, 0.2),
    }
    for method, weights in expected.items():
        cfg = load_configuration(ROOT, "casia_b_pose", method, "benchmark_v2_factorial")
        assert cfg["experiment_id"] == "benchmark_v2_factorial"
        assert cfg["model"]["backbone"] == "multistream"
        assert cfg["train"]["gradient_diagnostics_every_epochs"] == 5
        assert (cfg["method"].get("global_weight", 0), cfg["method"].get("part_weight", 0)) == weights


def test_aggregation_separates_backbones_and_does_not_count_duplicate_seeds(tmp_path):
    base = {"dataset": "casia_b_pose", "method": "oap_supcon", "seed": 11, "git_commit": "abc",
            "epochs": 1, "parameters": 8, "device": "cpu", "train_hours": .1, "status": "complete",
            "results": [{"protocol": "clean_gallery", "descriptor": "global", "corruption": "body_part",
                         "severity": 0, "rank1": .5}]}
    for name, backbone in [("a", "tcn"), ("b", "tcn"), ("c", "stgcn")]:
        run = tmp_path / name; run.mkdir()
        (run / "metrics.json").write_text(json.dumps({**base, "run_id": name, "backbone": backbone}))
    with pytest.warns(UserWarning, match="repeated-seed"):
        rows, summaries = aggregate_results(tmp_path)
    assert len(rows) == len(summaries) == 2
    assert all(r["rank1_count"] == 1 for r in summaries)
    with pytest.raises(ValueError, match="duplicate seed"):
        aggregate_results(tmp_path, duplicates="error")


def test_end_to_end_development_saves_checkpoint_and_never_scores_test(tmp_path, monkeypatch):
    path = make_smoke_dataset(tmp_path / "data/smoke/processed/dataset.npz")
    cfg = load_configuration(ROOT, "smoke", "oap_v2", "benchmark_v2")
    cfg["model"].update(hidden_dim=8, embedding_dim=16, options={"blocks": 2})
    cfg["train"].update(num_workers=0, identities_per_batch=3, samples_per_identity=2, clip_length=12, amp=False)
    cfg["train"]["gradient_diagnostics_every_epochs"] = 1
    cfg["validation"].update(every_epochs=1, corruption_families=["body_part"])
    cfg["evaluation"].update(corruption_families=["body_part"], severities=[0, .3], gallery_protocols=["clean"])
    # Every evaluated index must belong to the held-out training identities.
    import oap_supcon.experiment as module
    original = module._embeddings
    def guarded(model, data, indices, *args, **kwargs):
        assert np.isin(data.split[indices], ["val_gallery", "val_probe"]).all()
        return original(model, data, indices, *args, **kwargs)
    monkeypatch.setattr(module, "_embeddings", guarded)
    monkeypatch.delenv("OAP_DATA_ROOT", raising=False)
    run, metrics = run_experiment(tmp_path, cfg, 11, 2, "cpu", 1)
    assert metrics["evaluation_split"] == "validation"
    assert 1 <= metrics["selected_epoch"] <= 2
    for name in ["checkpoint.pt", "checkpoint-best.pt", "checkpoint-last.pt", "history.json", "probe_outcomes.npz"]:
        assert (run / name).exists()
    saved = torch.load(run / "checkpoint.pt", weights_only=True)
    assert saved["config"]["evaluation"]["corruption_realizations"] == 1
    assert len(saved["config"]["train_identities"]) == 6
    history = json.loads((run / "history.json").read_text())
    assert all(np.isfinite(row["loss"]) for row in history)
    assert all("validation" in row for row in history)
    assert all(len(row["part_reliability"]) == 5 for row in history)
    assert all(len(row["part_presence_rate"]) == 5 for row in history)
    assert all(len(row["part_embedding_variance"]) == 5 for row in history)
    assert all("loss_gradient_cosine" in row for row in history)
    best = max(history, key=lambda row: row["validation"]["score"])
    assert saved["epoch"] == best["epoch"]
    selected = torch.load(run / "checkpoint-best.pt", weights_only=True)
    assert all(torch.equal(saved["model"][key], selected["model"][key]) for key in saved["model"])


@pytest.mark.parametrize("method", ["ce", "masked_ce", "generic_supcon", "global_occlusion_supcon",
    "oap_supcon", "no_part", "no_temporal", "no_ce", "no_visibility", "no_complete_partial",
    "no_part_masking", "no_global", "oap_supcon_dropout", "matched_ce", "metric_baseline",
    "metric_global_supcon", "metric_part_supcon", "metric_part_triplet", "oap_v2", "oap_v2_ungated"])
def test_every_method_trains_and_evaluates_after_pipeline_change(tmp_path, monkeypatch, method):
    make_smoke_dataset(tmp_path / "data/smoke/processed/dataset.npz")
    cfg = load_configuration(ROOT, "smoke", method)
    cfg["model"].update(hidden_dim=8, embedding_dim=16)
    cfg["train"].update(identities_per_batch=3, samples_per_identity=2, clip_length=12)
    cfg["evaluation"].update(corruption_families=["body_part"], severities=[0, .3],
                             descriptors=["global", "reliable_parts"], gallery_protocols=["clean"])
    monkeypatch.delenv("OAP_DATA_ROOT", raising=False)
    run, metrics = run_experiment(tmp_path, cfg, 11, 1, "cpu", 1)
    assert metrics["status"] == "complete"
    assert all(np.isfinite(r["rank1"]) for r in metrics["results"])
    assert (run / "checkpoint.pt").exists()


def test_evaluation_failure_does_not_erase_trained_model(tmp_path, monkeypatch):
    make_smoke_dataset(tmp_path / "data/smoke/processed/dataset.npz")
    cfg = load_configuration(ROOT, "smoke", "ce")
    cfg["model"].update(hidden_dim=8, embedding_dim=16)
    cfg["train"].update(clip_length=12)
    import oap_supcon.experiment as module
    def fail(*args, **kwargs):
        raise RuntimeError("simulated evaluation failure")
    monkeypatch.setattr(module, "_evaluate", fail)
    monkeypatch.delenv("OAP_DATA_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="evaluation failure"):
        run_experiment(tmp_path, cfg, 11, 1, "cpu", 1)
    run = next((tmp_path / "results_v2").iterdir())
    assert (run / "checkpoint.pt").exists()
    assert (run / "checkpoint-last.pt").exists()
    assert (run / "history.json").exists()
    assert (run / "failure.json").exists()
    assert not (run / "metrics.json").exists()


@pytest.mark.parametrize("dimensions,joints", [(2, 18), (3, 17), (3, 25)])
def test_multistream_supports_other_representations(dimensions, joints):
    model = build_encoder("multistream", dimensions, joints, 3, 8, 16, 0, blocks=2)
    x = torch.randn(2, 12, joints, dimensions)
    v = torch.rand(2, 12, joints)
    v[:, 3:5] = 0
    out = model(x, v)
    assert out["parts"].shape == (2, 5, 16)
    loss = out["embedding"].square().mean() + out["logits"].mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
