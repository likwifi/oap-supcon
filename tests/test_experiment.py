import json
import math

import numpy as np
import pytest
import torch

from oap_supcon.experiment import retrieval_metrics
from oap_supcon.model import build_encoder


def test_retrieval_protocol_excludes_forbidden_gallery_pairs():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery_labels = np.array([0, 1])
    probe = np.array([[1.0, 0.0]], dtype=np.float32)
    probe_labels = np.array([0])
    unrestricted, _ = retrieval_metrics(gallery, gallery_labels, probe, probe_labels)
    restricted, _ = retrieval_metrics(
        gallery,
        gallery_labels,
        probe,
        probe_labels,
        exclusion_mask=np.array([[True, False]]),
    )
    assert unrestricted["rank1"] == 1.0
    assert restricted["rank1"] == 0.0


def test_part_reliability_is_normalised_over_real_frames_not_padding():
    # Regression: omega averaged over the padded buffer, so a fully visible part on a
    # 25-of-100-frame clip scored 0.25 and the part loss was scaled to ~6% of nominal.
    import torch

    from oap_supcon.augment import frame_mask_from_lengths
    from oap_supcon.model import PoseEncoder

    padded, real = 100, 25
    x = torch.zeros(1, padded, 17, 2)
    visibility = torch.zeros(1, padded, 17)
    x[:, :real] = 1.0
    visibility[:, :real] = 1.0
    lengths = torch.tensor([real])

    model = PoseEncoder(2, 17, 4).eval()
    frame_mask = frame_mask_from_lengths(visibility, lengths)
    with torch.no_grad():
        gated = model(x, visibility, frame_mask)["reliability"]
        ungated = model(x, visibility)["reliability"]
    assert torch.allclose(gated, torch.ones_like(gated), atol=1e-5)
    assert float(ungated.mean()) < 0.3  # the old, padding-dominated behaviour


def test_part_descriptor_is_unit_norm_and_distinct_from_the_global_one():
    """Retrieval must be able to use the part heads, not only the global embedding."""
    from oap_supcon.experiment import GLOBAL_DESCRIPTOR, PART_DESCRIPTOR, _descriptors

    torch.manual_seed(0)
    model = build_encoder("tcn", 2, 17, 4, 8, 16, 0.0).eval()
    x, visibility = torch.randn(3, 12, 17, 2), torch.ones(3, 12, 17)
    with torch.no_grad():
        descriptors = _descriptors(model(x, visibility))
    assert set(descriptors) == {GLOBAL_DESCRIPTOR, PART_DESCRIPTOR}
    combined = descriptors[PART_DESCRIPTOR]
    assert combined.shape == (3, 16 * 6)
    assert torch.allclose(combined.norm(dim=-1), torch.ones(3), atol=1e-5)
    # The global block is the global embedding, rescaled by the same factor.
    assert torch.allclose(
        combined[:, :16] * math.sqrt(6), descriptors[GLOBAL_DESCRIPTOR], atol=1e-5
    )


def test_raw_part_descriptors_are_available_without_matching_vector_widths():
    """Raw matching is a normal evaluator option, not a validation-only control."""
    from oap_supcon.experiment import (
        RAW_RELIABLE_DESCRIPTOR,
        RAW_UNIFORM_DESCRIPTOR,
        _descriptors,
    )

    torch.manual_seed(0)
    # TCN raw pools have hidden_dim=8 while the global descriptor has width 16.
    model = build_encoder("tcn", 2, 17, 4, 8, 16, 0.0).eval()
    x, visibility = torch.randn(3, 12, 17, 2), torch.ones(3, 12, 17)
    visibility[0, :, [5, 7, 9]] = 0
    with torch.no_grad():
        output = model(x, visibility)
        descriptors = _descriptors(output, include_raw=True)
    reliable = descriptors[RAW_RELIABLE_DESCRIPTOR]
    uniform = descriptors[RAW_UNIFORM_DESCRIPTOR]
    assert reliable.shape == uniform.shape == (3, 6, 17)
    assert torch.equal(reliable[:, 1:, -1], output["reliability"])
    assert torch.equal(uniform[:, 1:, -1], (output["reliability"] > 0).float())
    assert reliable[0, 2, -1] == 0


def test_clean_rank1_table_uses_the_overall_row_not_a_condition_slice(tmp_path):
    """`official_*` rows also carry severity 0 and are emitted first."""
    from oap_supcon.reporting import aggregate_results

    run = tmp_path / "casia_b_pose_ce_seed11_x"
    run.mkdir()
    common = {"rank5": 0.0, "rank10": 0.0, "map": 0.0, "eer": 0.0}
    (run / "metrics.json").write_text(json.dumps({
        "run_id": run.name, "git_commit": "abc", "dataset": "casia_b_pose", "method": "ce",
        "seed": 11, "epochs": 1, "device": "cpu", "parameters": 1, "train_hours": 0.1,
        "status": "complete",
        "results": [
            {"protocol": "clean_gallery", "descriptor": "global",
             "corruption": "official_condition:BG", "severity": 0.0, "rank1": 0.99, **common},
            {"protocol": "clean_gallery", "descriptor": "global",
             "corruption": "body_part", "severity": 0.0, "rank1": 0.25, **common},
        ],
    }))
    aggregate_results(tmp_path)
    table = (tmp_path / "clean_rank1.tex").read_text()
    assert "0.250" in table
    assert "0.990" not in table, "the BG condition row was reported as clean rank-1"


def test_single_view_gallery_excludes_the_identical_view_diagonal():
    """Published CASIA-B numbers come from this protocol, not the pooled gallery."""
    from oap_supcon.experiment import _single_view_gallery

    # Identity 1 is recognisable only within a view, identity 2 only across
    # views. Pooling the gallery would score both; the official protocol drops
    # the identical-view cells, so only identity 2 can be matched.
    gallery = {"global": np.array([[1, 0], [0, 1], [1, 0], [1, 0]], np.float32)}
    gallery_labels, gallery_views = np.array([1, 2, 1, 2]), np.array(["000", "000", "090", "090"])
    probe = {"global": np.array([[1, 0], [1, 0]], np.float32)}
    probe_labels, probe_views = np.array([1, 1]), np.array(["000", "090"])
    probe_conditions = np.array(["NM", "NM"])

    rows = _single_view_gallery(
        gallery, gallery_labels, gallery_views,
        probe, probe_labels, probe_conditions, probe_views, ["global"],
    )
    assert len(rows) == 3
    row = next(r for r in rows if r["protocol"] == "single_view_gallery")
    assert row["protocol"] == "single_view_gallery"
    assert row["corruption"] == "official_condition:NM"
    # probe(000) vs gallery(090) picks label 1; probe(090) vs gallery(000) picks
    # label 1 as well -- both off-diagonal cells are correct.
    assert row["rank1"] == 1.0
    per_view = [r for r in rows if r["protocol"] == "single_view_gallery_per_view"]
    assert {r["probe_view"] for r in per_view} == {"000", "090"}
    assert all(r["rank1"] == 1.0 for r in per_view)


def test_single_view_gallery_needs_more_than_one_view():
    from oap_supcon.experiment import _single_view_gallery

    single = {"global": np.array([[1.0, 0.0]], np.float32)}
    with pytest.raises(ValueError, match="at least two camera views"):
        _single_view_gallery(
            single, np.array([1]), np.array(["000"]),
            single, np.array([1]), np.array(["NM"]), np.array(["000"]), ["global"],
        )


def test_eer_and_tar_match_the_threshold_sweep_definition():
    """The binary-search counts must not change the reported operating points."""
    rng = np.random.default_rng(7)
    gallery = rng.normal(size=(40, 8)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    probe = rng.normal(size=(60, 8)).astype(np.float32)
    probe /= np.linalg.norm(probe, axis=1, keepdims=True)
    gallery_labels, probe_labels = rng.integers(0, 5, 40), rng.integers(0, 5, 60)
    exclusion = rng.random((60, 40)) < 0.2
    exclusion[(~exclusion).sum(axis=1) == 0, 0] = False

    metrics, _ = retrieval_metrics(gallery, gallery_labels, probe, probe_labels, exclusion)

    similarity = probe @ gallery.T
    similarity[exclusion] = -np.inf
    genuine = similarity[(probe_labels[:, None] == gallery_labels[None, :]) & ~exclusion]
    impostor = similarity[(probe_labels[:, None] != gallery_labels[None, :]) & ~exclusion]
    thresholds = np.quantile(np.concatenate([genuine, impostor]), np.linspace(0, 1, 2001))
    false_accept = np.array([(impostor >= t).mean() for t in thresholds])
    false_reject = np.array([(genuine < t).mean() for t in thresholds])
    index = int(np.argmin(np.abs(false_accept - false_reject)))

    assert metrics["eer"] == pytest.approx(
        (false_accept[index] + false_reject[index]) / 2, abs=1e-12
    )
    for far in (1e-2, 1e-3):
        threshold = np.quantile(impostor, 1 - far, method="higher")
        assert metrics[f"tar_far_{far:.0e}"] == pytest.approx(
            (genuine >= threshold).mean(), abs=1e-12
        )


def _tiny_two_view_dataset():
    from oap_supcon.data import PoseData

    rng = np.random.default_rng(1)
    views, conditions, labels, splits = [], [], [], []
    for identity in range(1, 5):
        for view in ("000", "090"):
            for split, condition in (("gallery", "NM"), ("probe", "NM"), ("probe", "BG")):
                views.append(view)
                conditions.append(condition)
                labels.append(identity)
                splits.append(split)
    # Train identities must be disjoint from the test ones.
    for identity in range(10, 14):
        for view in ("000", "090"):
            views.append(view)
            conditions.append("NM")
            labels.append(identity)
            splits.append("train")
    count = len(labels)
    x = rng.normal(size=(count, 12, 17, 2)).astype(np.float32)
    return PoseData(
        x, np.ones((count, 12, 17), np.float32), np.array(labels), np.array(splits),
        np.array([str(i) for i in range(count)]), np.array(conditions), np.array(views),
        np.full(count, 12, np.int32),
    )


def test_evaluate_emits_official_protocol_rows_when_the_dataset_declares_it():
    """The flag lives in the dataset config; reading it elsewhere silently skips it."""
    from oap_supcon.experiment import _evaluate

    data = _tiny_two_view_dataset()
    torch.manual_seed(0)
    model = build_encoder("tcn", 2, 17, 4, 8, 16, 0.0).eval()
    cfg = {
        "dataset": {"exclude_identical_view": True, "single_view_gallery": True},
        "pose": {"weight_coordinates": False},
        "train": {"clip_length": None},
        "evaluation": {
            "corruption_families": ["random_joint"], "severities": [0.0],
            "gallery_protocols": ["clean"],
            "descriptors": ["global", "global_parts", "raw_reliable", "raw_uniform"],
            "clip_length": None,
        },
    }
    rows, _, _ = _evaluate(model, data, cfg, torch.device("cpu"), seed=0, realizations=1)
    official = [r for r in rows if r["protocol"] == "single_view_gallery"]
    assert {r["corruption"] for r in official} == {
        "official_condition:NM", "official_condition:BG"
    }
    assert {r["descriptor"] for r in official} == {
        "global", "global_parts", "raw_reliable", "raw_uniform"
    }
    assert all(0.0 <= r["rank1"] <= 1.0 for r in official)
    per_view = [r for r in rows if r["protocol"] == "single_view_gallery_per_view"]
    assert len(per_view) == 4 * 2 * 2  # descriptors x conditions x probe views


def test_aggregation_exports_official_per_view_rows(tmp_path):
    from oap_supcon.reporting import aggregate_results

    run = tmp_path / "run"
    run.mkdir()
    common = {
        "run_id": "run", "git_commit": "abc", "dataset": "casia_b_pose",
        "method": "metric_baseline", "backbone": "multistream",
        "experiment_id": "benchmark_v2", "config_fingerprint": "cfg",
        "source_sha256": "src", "dataset_sha256": "data",
        "evaluation_split": "test", "seed": 11, "epochs": 1,
        "device": "cpu", "parameters": 1, "train_hours": 0.1,
        "status": "complete",
    }
    (run / "metrics.json").write_text(json.dumps({
        **common,
        "results": [{
            "protocol": "single_view_gallery_per_view", "descriptor": "raw_reliable",
            "corruption": "official_condition:NM", "probe_view": "090",
            "severity": 0.0, "rank1": 0.75,
        }],
    }))
    aggregate_results(tmp_path)
    exported = (tmp_path / "official_per_view.csv").read_text()
    assert "condition,probe_view,rank1_mean" in exported
    assert "NM,090,0.75" in exported


def test_evaluate_skips_the_official_protocol_when_the_dataset_does_not_declare_it():
    from oap_supcon.experiment import _evaluate

    data = _tiny_two_view_dataset()
    torch.manual_seed(0)
    model = build_encoder("tcn", 2, 17, 4, 8, 16, 0.0).eval()
    cfg = {
        "dataset": {"exclude_identical_view": True},
        "pose": {"weight_coordinates": False},
        "train": {"clip_length": None},
        "evaluation": {
            "corruption_families": ["random_joint"], "severities": [0.0],
            "gallery_protocols": ["clean"], "descriptors": ["global"],
            "clip_length": None,
        },
    }
    rows, _, _ = _evaluate(model, data, cfg, torch.device("cpu"), seed=0, realizations=1)
    assert not [r for r in rows if r["protocol"] == "single_view_gallery"]
