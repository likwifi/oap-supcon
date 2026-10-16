import csv
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import yaml

from oap_supcon.data import PoseData


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTORS = ("global", "raw_reliable", "raw_uniform")
# Three skeleton heights x six probes. Probe k of each height has 5k low-confidence
# joint-frames, so both global and within-height tertiles are {0,1}, {2,3}, {4,5}.
LEVELS = np.tile(np.arange(6), 3)
CLEAN = {"global": (LEVELS < 4).astype(float), "raw_reliable": np.ones(18),
         "raw_uniform": (LEVELS < 5).astype(float)}


def _dataset(path):
    frames, joints = 6, 17
    labels, split, confidence, poses = [], [], [], []
    rng = np.random.default_rng(0)

    def pose(scale):
        x = rng.normal(size=(frames, joints, 2))
        x[..., 1] = np.linspace(0, scale, joints)
        return x

    for identity in range(3):
        labels += [identity, identity]
        split += ["train", "train"]
        confidence += [np.full((frames, joints), .9)] * 2
        poses += [pose(100), pose(100)]
    for number, level in enumerate(LEVELS):
        identity = 3 + number
        labels += [identity, identity]
        split += ["val_gallery", "val_probe"]
        probe = np.full(frames * joints, .9)
        probe[:5 * level] = .2
        confidence += [np.full((frames, joints), .9), probe.reshape(frames, joints)]
        poses += [pose(100), pose(100 * (1 + number // 6))]
    count = len(labels)
    np.savez(path, x=np.stack(poses).astype(np.float32),
             visibility=np.stack(confidence).astype(np.float32), labels=np.array(labels),
             split=np.array(split), sequence_ids=np.array([f"{l:04d}-camid0_videoid1-seq{i}"
                                                         for i, l in enumerate(labels)]),
             views=np.full(count, "camid0"), conditions=np.full(count, "in_the_wild"),
             frame_counts=np.full(count, frames, dtype=np.int32))
    return PoseData.load(path)


def _runs(root, dataset_path, data, seeds=(11, 22)):
    probes = data.indices(["val_probe"])
    for seed in seeds:
        directory = root / f"run{seed}"
        directory.mkdir(parents=True)
        config = {"dataset": {"name": "gait3d"}, "method_name": "metric_baseline",
                  "validation": {"enabled": True, "holdout_identities": 18, "split_seed": 2026},
                  "evaluation": {"descriptors": list(DESCRIPTORS), "part_weight": .5},
                  "dataset_path": str(dataset_path)}
        (directory / "config.yaml").write_text(yaml.safe_dump(config))
        arrays = {"probe_indices": probes, "probe_labels": data.labels[probes]}
        for descriptor in DESCRIPTORS:
            outcome = np.asarray(CLEAN[descriptor], dtype=np.float32)
            arrays[f"{descriptor}_clean_gallery_body_part_0"] = outcome
            arrays[f"{descriptor}_clean_gallery_temporal_0"] = outcome
            arrays[f"{descriptor}_clean_gallery_body_part_0.3"] = np.zeros(18, dtype=np.float32)
        np.savez_compressed(directory / "probe_outcomes.npz", **arrays)
        metrics = {"run_id": directory.name, "seed": seed, "status": "complete",
                   "dataset_sha256": data.checksum(), "source_sha256": "source",
                   "config_fingerprint": "recipe", "evaluation_split": "validation",
                   "backbone": "multistream", "results": []}
        (directory / "metrics.json").write_text(json.dumps(metrics))


@pytest.fixture
def analyzer():
    return runpy.run_path(str(ROOT / "scripts/analyze_gait3d_occlusion.py"))


def test_strata_contrasts_and_primary_interaction(tmp_path, analyzer):
    path = tmp_path / "dataset.npz"
    data = _dataset(path)
    _runs(tmp_path / "runs", path, data)
    output = tmp_path / "out"
    result = analyzer["analyze"](tmp_path / "runs", output, [11, 22], 200, 0, audit_per_bin=1)
    assert result["probes"] == 18 and result["evaluation_split"] == "validation"
    assert result["primary"]["scope"] == "height_matched_interaction_high_minus_low"
    assert result["primary"]["difference"] == pytest.approx(1.0)
    assert result["secondary"]["scope"] == "interaction_high_minus_low"
    assert result["secondary"]["difference"] == pytest.approx(1.0)
    with (output / "occlusion_strata.csv").open() as handle:
        strata = {(r["proxy"], r["stratum"], r["descriptor"]): r for r in csv.DictReader(handle)}
    primary = "low_confidence_fraction"
    assert float(strata[(primary, "low", "global")]["rank1"]) == 1
    assert float(strata[(primary, "high", "global")]["rank1"]) == 0
    assert float(strata[(primary, "high", "raw_uniform")]["rank1"]) == .5
    assert float(strata[(primary, "all", "raw_reliable")]["rank1"]) == 1
    assert int(strata[(primary, "middle", "global")]["probes"]) == 6
    with (output / "occlusion_contrasts.csv").open() as handle:
        contrasts = list(csv.DictReader(handle))
    effect = next(r for r in contrasts if r["proxy"] == primary and r["contrast"] == "global_high_minus_low"
                  and r["scope"] == "height_matched_occlusion_effect")
    assert float(effect["difference"]) == -1
    assert [r["role"] for r in contrasts].count("primary") == 1
    with (output / "manual_audit_sheet.csv").open() as handle:
        sheet = list(csv.DictReader(handle))
    assert len(sheet) == 3 and "primary_bin" not in sheet[0] and primary not in sheet[0]
    assert (output / "manual_audit_key.csv").exists()
    assert (output / "probe_occlusion_proxies.csv").exists()


def test_run_selection_skips_failed_and_duplicate_attempts(tmp_path, analyzer):
    path = tmp_path / "dataset.npz"
    data = _dataset(path)
    root = tmp_path / "runs"
    _runs(root, path, data)
    failed = root / "failed11"
    failed.mkdir()
    (failed / "config.yaml").write_text((root / "run11/config.yaml").read_text())
    with pytest.raises(ValueError, match="incomplete run"):
        analyzer["analyze"](root, tmp_path / "out", [11, 22], 50, 0, audit_per_bin=0)
    result = analyzer["analyze"](root, tmp_path / "out", [11, 22], 50, 0, audit_per_bin=0,
                                 run_names=["run11", "run22"])
    assert result["runs"] == {"11": "run11", "22": "run22"}


def test_mismatched_data_or_probe_order_is_rejected(tmp_path, analyzer):
    path = tmp_path / "dataset.npz"
    data = _dataset(path)
    root = tmp_path / "runs"
    _runs(root, path, data)
    archive_path = root / "run22/probe_outcomes.npz"
    with np.load(archive_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["probe_indices"] = arrays["probe_indices"][::-1]
    np.savez_compressed(archive_path, **arrays)
    with pytest.raises(ValueError, match="probes differ"):
        analyzer["analyze"](root, tmp_path / "out", [11, 22], 50, 0, audit_per_bin=0)
    for seed in (11, 22):
        metrics_path = root / f"run{seed}/metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["dataset_sha256"] = "other"
        metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="does not match"):
        analyzer["analyze"](root, tmp_path / "out", [11, 22], 50, 0, audit_per_bin=0)


def test_quantile_bins_keep_ties_together_and_reject_empty_bins(analyzer):
    assignment, _ = analyzer["quantile_bins"](np.array([0., .1, .2, .3, .4, .5]), 3)
    assert assignment.tolist() == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError, match="empty quantile bin"):
        analyzer["quantile_bins"](np.array([0., 0, 0, 0, .1, .2]), 3)
    assert analyzer["spearman"](np.array([1., 2, 3]), np.array([3., 5, 9])) == pytest.approx(1)
