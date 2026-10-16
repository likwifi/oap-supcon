import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from oap_supcon.data import PoseData, development_split
from oap_supcon.experiment import _evaluate, load_configuration, retrieval_metrics, run_experiment
from oap_supcon.gait3d import convert_gait3d_pose, dataset_coverage, load_partition, load_pose_frame
from oap_supcon.model import build_encoder
from oap_supcon.reporting import audit_dataset


ROOT = Path(__file__).resolve().parents[1]


def _frame(path, offset=0, confidence=.8):
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = np.column_stack((np.arange(17) + offset, np.arange(17) * 2, np.full(17, confidence)))
    np.savetxt(path, np.concatenate(([100, 200], pose.ravel()))[None], delimiter=",")


def _release(tmp_path):
    source = tmp_path / "raw"
    partition = {"TRAIN_SET": ["0000", "0001", "0002", "0003"],
                 "TEST_SET": ["0004", "0005"],
                 "PROBE_SET": [f"{i:04d}-camid0_videoid1-seq1" for i in (4, 5)]}
    for identity in range(6):
        for seq in range(2):
            directory = source / f"{identity:04d}" / "camid0_videoid1" / f"seq{seq}"
            _frame(directory / "human_crop_f10.txt", offset=10, confidence=1.03)
            _frame(directory / "human_crop_f2.txt", offset=2)
            if seq == 0:
                _frame(directory / "human_crop_f11.txt", offset=11)
    (source / ".DS_Store").write_text("ignored")
    path = tmp_path / "Gait3D.json"
    path.write_text(json.dumps(partition))
    return source, path


def test_conversion_preserves_order_confidence_lengths_and_official_roles(tmp_path):
    source, partition = _release(tmp_path)
    destination = tmp_path / "dataset.npz"
    report = convert_gait3d_pose(source, partition, destination, progress=None)
    data = PoseData.load(destination)
    assert report["complete"]
    assert report["split_sequences"] == {"train": 8, "gallery": 2, "probe": 2}
    assert data.x.shape == (12, 3, 17, 2)
    assert data.x[0, :, 0, 0].tolist() == [2, 10, 11]
    assert data.frame_counts.tolist() == [3, 2] * 6
    assert np.all(data.x[1, 2] == 0) and np.all(data.visibility[1, 2] == 0)
    assert np.all(data.visibility[:, 1] == 1)
    assert report["confidence_scores_clipped"] == 12 * 17
    assert data.views.tolist() == ["camid0"] * 12
    assert data.sequence_ids[-1] == "0005-camid0_videoid1-seq1"
    with np.load(destination, allow_pickle=False) as archive:
        assert archive["protocol"].item() == "gait3d_official"
        assert json.loads(archive["coverage_json"].item())["dataset_sha256"] == data.checksum()
    with pytest.raises(FileExistsError):
        convert_gait3d_pose(source, partition, destination, progress=None)


def test_incomplete_release_is_reported_and_requires_explicit_subset_export(tmp_path):
    source, partition = _release(tmp_path)
    for path in (source / "0004" / "camid0_videoid1" / "seq0").glob("*.txt"):
        path.unlink()
    (source / "0004" / "camid0_videoid1" / "seq0").rmdir()
    destination, coverage = tmp_path / "dataset.npz", tmp_path / "coverage.json"
    with pytest.raises(ValueError, match="full partition"):
        convert_gait3d_pose(source, partition, destination, report_path=coverage, progress=None)
    assert not destination.exists()
    assert json.loads(coverage.read_text())["probes_without_gallery"] == ["0004-camid0_videoid1-seq1"]
    report = convert_gait3d_pose(source, partition, destination, allow_incomplete=True, progress=None)
    data = PoseData.load(destination)
    assert report["protocol"] == "gait3d_available_pose"
    with pytest.raises(ValueError, match="does not match"):
        dataset_coverage(destination)
    assert not dataset_coverage(destination, "gait3d_available_pose")["complete"]
    assert data.labels[data.split == "probe"].tolist() == [5]
    audit_dataset(destination, tmp_path / "audit")
    assert "Excluded probes without gallery: 1" in (tmp_path / "audit/data_audit.md").read_text()


@pytest.mark.parametrize("row", ["1,2,3", ",".join(["1"] * 53) + ",garbage",
                                  ",".join(["nan"] * 53)])
def test_malformed_pose_rows_are_rejected(tmp_path, row):
    path = tmp_path / "human_crop_f1.txt"
    path.write_text(row)
    with pytest.raises(ValueError):
        load_pose_frame(path)


def test_partition_rejects_identity_leakage_and_wrong_probe_identity(tmp_path):
    _, path = _release(tmp_path)
    partition = json.loads(path.read_text())
    partition["TEST_SET"].append("0000")
    path.write_text(json.dumps(partition))
    with pytest.raises(ValueError, match="disjoint"):
        load_partition(path)


def test_gait3d_validation_is_reproducible_and_final_restores_all_training_ids(tmp_path):
    source, partition = _release(tmp_path)
    path = tmp_path / "dataset.npz"
    convert_gait3d_pose(source, partition, path, progress=None)
    data = PoseData.load(path)
    cfg = {"dataset": {"name": "gait3d"},
           "validation": {"enabled": True, "holdout_identities": 2, "split_seed": 2026}}
    dev, held = development_split(data, cfg)
    repeat, repeat_held = development_split(data, cfg)
    assert held == repeat_held and np.array_equal(dev.split, repeat.split)
    assert dev.x is data.x
    for identity in held:
        assert np.sum((dev.labels == identity) & (dev.split == "val_probe")) == 1
        assert np.sum((dev.labels == identity) & (dev.split == "val_gallery")) == 1
    test = np.isin(data.split, ["gallery", "probe"])
    assert np.array_equal(dev.split[test], data.split[test])
    assert np.unique(dev.labels[dev.split == "train"]).size == 2
    cfg["validation"]["enabled"] = False
    final, final_held = development_split(data, cfg)
    assert final_held == [] and final is data
    assert np.unique(final.labels[final.split == "train"]).size == 4
    # Probe choice must depend on immutable sequence IDs, not storage order.
    permutation = np.arange(len(data.labels))[::-1]
    reordered = replace(data, **{name: getattr(data, name)[permutation] for name in (
        "x", "visibility", "labels", "split", "sequence_ids", "conditions", "views", "frame_counts")})
    shuffled, _ = development_split(reordered, {**cfg, "validation": {**cfg["validation"], "enabled": True}})
    assert set(shuffled.sequence_ids[shuffled.split == "val_probe"]) == set(dev.sequence_ids[dev.split == "val_probe"])


def test_minp_uses_the_rank_of_the_last_relevant_gallery_sample():
    gallery = np.array([[1., 0], [.8, 0], [.6, 0], [.4, 0]], dtype=np.float32)
    metrics, _ = retrieval_metrics(gallery, np.array([0, 1, 1, 0]),
                                   np.array([[1., 0]], dtype=np.float32), np.array([0]))
    assert metrics["rank1"] == 1
    assert metrics["map"] == pytest.approx(.75)
    assert metrics["minp"] == pytest.approx(.5)


def test_gait3d_recipe_and_evaluator_support_all_frozen_descriptors(tmp_path):
    source, partition = _release(tmp_path)
    path = tmp_path / "dataset.npz"
    convert_gait3d_pose(source, partition, path, progress=None)
    cfg = load_configuration(ROOT, "gait3d", "metric_baseline", "gait3d_v2_available")
    assert cfg["dataset"]["file"] == "processed/dataset.npz"
    assert not cfg["dataset"]["exclude_identical_view"]
    assert cfg["evaluation"]["part_weight"] == .5
    cfg["validation"]["holdout_identities"] = 2
    cfg["evaluation"].update({"corruption_families": ["body_part"], "severities": [0., .3],
                              "gallery_protocols": ["clean"], "batch_size": 4})
    data, _ = development_split(PoseData.load(path), cfg)
    model = build_encoder("multistream", 2, 17, 2, 8, 16, 0, blocks=2).eval()
    rows, outcomes, _ = _evaluate(model, data, cfg, torch.device("cpu"), 11, 1)
    assert {r["descriptor"] for r in rows} == {"global", "raw_reliable", "raw_uniform"}
    assert all("minp" in row for row in rows)
    assert "raw_reliable_clean_gallery_body_part_0.3" in outcomes
    ap = outcomes["ap__raw_reliable_clean_gallery_body_part_0.3"]
    assert ap.shape == outcomes["raw_reliable_clean_gallery_body_part_0.3"].shape
    assert np.all((ap >= 0) & (ap <= 1))
    row = next(r for r in rows if r["descriptor"] == "raw_reliable" and r["protocol"] == "clean_gallery"
               and r["corruption"] == "body_part" and r["severity"] == .3)
    assert row["map"] == pytest.approx(float(ap.mean()), abs=1e-6)


def test_gait3d_training_writes_coverage_and_matched_descriptor_artifacts(tmp_path, monkeypatch):
    source, partition = _release(tmp_path)
    path = tmp_path / "data/gait3d/processed/dataset.npz"
    convert_gait3d_pose(source, partition, path, progress=None)
    monkeypatch.delenv("OAP_DATA_ROOT", raising=False)
    cfg = load_configuration(ROOT, "gait3d", "metric_baseline", "gait3d_v2")
    cfg["model"].update({"backbone": "tcn", "hidden_dim": 8, "embedding_dim": 16, "options": {}})
    cfg["validation"].update({"holdout_identities": 2, "every_epochs": 1})
    cfg["train"].update({"num_workers": 0, "clip_length": 4, "identities_per_batch": 2,
                          "samples_per_identity": 2, "amp": False})
    cfg["evaluation"].update({"corruption_families": ["body_part"], "severities": [0., .3],
                              "gallery_protocols": ["clean"], "batch_size": 4})
    directory, metrics = run_experiment(tmp_path, cfg, 11, 1, "cpu", 1)
    assert metrics["source_coverage"]["complete"]
    assert metrics["evaluation_split"] == "validation"
    assert (directory / "checkpoint.pt").exists()
    assert (directory / "probe_outcomes.npz").exists()
    assert {r["descriptor"] for r in metrics["results"]} == set(DESCRIPTORS)


DESCRIPTORS = ("global", "raw_reliable", "raw_uniform")


def _comparison_runs(root):
    for seed in (11, 22):
        directory = root / f"run{seed}"
        directory.mkdir(parents=True)
        config = {"dataset": {"name": "gait3d"}, "method_name": "metric_baseline",
                  "model": {"backbone": "tcn"}, "evaluation": {
                      "descriptors": list(DESCRIPTORS), "part_weight": .5}}
        (directory / "config.yaml").write_text(yaml.safe_dump(config))
        rows, arrays = [], {"probe_indices": np.array([1, 3]), "probe_labels": np.array([4, 5])}
        for descriptor in DESCRIPTORS:
            outcomes = np.array([1., 1.]) if descriptor == "raw_reliable" else np.array([1., 0.])
            for severity in (0., .3):
                rows.append({"descriptor": descriptor, "protocol": "clean_gallery",
                             "corruption": "body_part", "severity": severity,
                             **{metric: float(outcomes.mean()) for metric in ("rank1", "rank5", "rank10", "map", "minp")}})
                arrays[f"{descriptor}_clean_gallery_body_part_{severity:g}"] = outcomes
        metrics = {"run_id": directory.name, "dataset": "gait3d", "method": "metric_baseline",
                   "seed": seed, "status": "complete", "dataset_sha256": "data", "source_sha256": "source",
                   "config_fingerprint": "recipe", "evaluation_split": "validation", "backbone": "tcn",
                   "git_commit": "test", "epochs": 1, "device": "cpu", "parameters": 1,
                   "train_hours": .1, "results": rows}
        (directory / "metrics.json").write_text(json.dumps(metrics))
        np.savez_compressed(directory / "probe_outcomes.npz", **arrays)


def test_descriptor_comparison_pairs_seeds_and_rejects_different_data(tmp_path):
    analyzer = runpy.run_path(str(ROOT / "scripts/analyze_gait3d.py"))
    root = tmp_path / "runs"
    _comparison_runs(root)
    result = analyzer["analyze"](root, tmp_path / "summary", [11, 22], 100, 0)
    assert result["paired"]["raw_reliable_minus_global:clean"]["difference"] == .5
    assert result["paired"]["raw_reliable_minus_raw_uniform:robust"]["difference"] == .5
    assert (tmp_path / "summary/descriptor_comparison.csv").exists()
    assert (tmp_path / "summary/summary.csv").exists()
    path = root / "run22/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["dataset_sha256"] = "other-data"
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="different data"):
        analyzer["discover_runs"](root, [11, 22])
