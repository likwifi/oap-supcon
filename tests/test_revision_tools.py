import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from oap_supcon.data import PoseData
from oap_supcon.experiment import AP_PREFIX, _embeddings, retrieval_metrics
from oap_supcon.gait3d import convert_gait3d_pose
from oap_supcon.reference_baseline import prepare_fastposegait

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bootstrap_stats import crossed_draws  # noqa: E402

DESCRIPTORS = ("global", "raw_reliable", "raw_uniform")


def _script(name):
    return runpy.run_path(str(ROOT / "scripts" / name))


def test_crossed_bootstrap_shares_identity_draws_across_seeds():
    # Identity dominates: every seed agrees on each identity, so the interval must
    # reflect between-identity spread rather than shrinking it by averaging seeds.
    identity_effect = np.linspace(-1, 1, 12)
    values = np.tile(identity_effect, (5, 1))
    crossed = crossed_draws(values, 4000, np.random.default_rng(0))
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 5, size=(4000, 5))
    nested = np.take_along_axis(values[seeds], rng.integers(0, 12, size=(4000, 5, 12)), axis=2).mean(axis=(1, 2))
    assert crossed.std() > 2 * nested.std()
    assert crossed.std() == pytest.approx(identity_effect.std() / np.sqrt(12), rel=.1)
    with pytest.raises(ValueError):
        crossed_draws(np.zeros(3), 10, rng)


def test_retrieval_metrics_returns_per_probe_average_precision():
    gallery = np.array([[1., 0], [.8, .6], [0, 1.]], dtype=np.float32)
    probe = np.array([[1., 0], [0, 1.]], dtype=np.float32)
    metrics, correct, ap = retrieval_metrics(gallery, np.array([0, 1, 1]), probe, np.array([1, 1]), return_ap=True)
    assert correct.tolist() == [0, 1]
    assert ap.tolist() == pytest.approx([(1 / 2 + 2 / 3) / 2, 1.0])
    assert metrics["map"] == pytest.approx(ap.mean())


def _cohort(root, methods=("ce", "metric_baseline"), seeds=(11, 22), with_ap=True, source="s"):
    labels = np.array([5, 5, 6, 7])
    families, severities = ["body_part"], [0.0, .5]
    for method in methods:
        for seed in seeds:
            directory = root / f"{method}_{seed}"
            directory.mkdir(parents=True)
            config = {"dataset": {"name": "toy"}, "method_name": method,
                      "evaluation": {"descriptors": list(DESCRIPTORS), "corruption_families": families,
                                     "severities": severities, "gallery_protocols": ["clean"]}}
            (directory / "config.yaml").write_text(yaml.safe_dump(config))
            arrays = {"probe_indices": np.arange(4), "probe_labels": labels}
            for descriptor in DESCRIPTORS:
                base = 1.0 if method == "metric_baseline" else 0.0
                bonus = .5 if descriptor != "global" else 0.0
                for severity in severities:
                    key = f"{descriptor}_clean_gallery_body_part_{severity:g}"
                    arrays[key] = np.full(4, min(1.0, base * (1 - severity) + bonus), dtype=np.float32)
                    if with_ap:
                        arrays[AP_PREFIX + key] = arrays[key] * .8
            np.savez_compressed(directory / "probe_outcomes.npz", **arrays)
            metrics = {"seed": seed, "status": "complete", "dataset_sha256": "d", "source_sha256": source,
                       "evaluation_split": "validation", "backbone": "multistream", "epochs": 1, "results": []}
            (directory / "metrics.json").write_text(json.dumps(metrics))


def test_compare_runs_reports_rank1_and_map_with_paired_contrasts(tmp_path):
    compare = _script("compare_runs.py")
    _cohort(tmp_path / "runs")
    result = compare["compare"](tmp_path / "runs", tmp_path / "out", "toy", ["ce", "metric_baseline"],
                                [11, 22], "metric_baseline", 200, 0)
    assert result["probes"] == 4 and result["identities"] == 3 and result["unavailable_metrics"] == []
    import csv
    with (tmp_path / "out/comparison_contrasts.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    method = {(r["descriptor"], r["endpoint"], r["metric"]): float(r["mean"])
              for r in rows if r["type"] == "method"}
    assert method[("global", "clean", "rank1")] == pytest.approx(-1.0)
    assert method[("global", "robust", "map")] == pytest.approx(-.4)
    descriptor = {(r["method"], r["endpoint"], r["metric"]): float(r["mean"])
                  for r in rows if r["type"] == "descriptor" and r["comparison"] == "raw_reliable_minus_global"}
    assert descriptor[("ce", "clean", "rank1")] == pytest.approx(.5)
    assert descriptor[("metric_baseline", "clean", "rank1")] == pytest.approx(0)


def test_compare_runs_rejects_mixed_source_and_flags_missing_ap(tmp_path):
    compare = _script("compare_runs.py")
    _cohort(tmp_path / "mixed", methods=("ce",), source="old")
    _cohort(tmp_path / "mixed", methods=("metric_baseline",), source="new")
    with pytest.raises(ValueError, match="source revision"):
        compare["compare"](tmp_path / "mixed", tmp_path / "o1", "toy", ["ce", "metric_baseline"],
                           [11, 22], "metric_baseline", 50, 0)
    _cohort(tmp_path / "legacy", with_ap=False)
    result = compare["compare"](tmp_path / "legacy", tmp_path / "o2", "toy", ["ce", "metric_baseline"],
                                [11, 22], "metric_baseline", 50, 0)
    assert result["unavailable_metrics"] == ["map"]


def test_epoch_budget_rule_distinguishes_plateau_peak_and_rising_curves():
    choose = _script("choose_epoch_budget.py")["choose"]
    epochs = list(range(5, 165, 5))

    def curves(shape):
        return {seed: {e: shape(e) + offset for e in epochs} for seed, offset in ((11, 0), (22, .01), (33, -.01))}

    plateau = choose(curves(lambda e: min(e, 100) / 300), 160, .01, 3)
    assert plateau["decision"] == "full_schedule" and plateau["budget"] == 160
    peaked = choose(curves(lambda e: .4 - abs(e - 80) / 400), 160, .01, 3)
    assert peaked["decision"] == "smoothed_peak" and 75 <= peaked["budget"] <= 85
    rising = choose(curves(lambda e: e / 500), 160, .01, 3)
    assert rising["decision"] == "extend_schedule" and rising["budget"] is None
    with pytest.raises(ValueError, match="schedule end"):
        choose(curves(lambda e: e / 500), 200, .01, 3)


def _casia_like(path):
    rng = np.random.default_rng(3)
    count, frames = 6, 9
    np.savez(path, x=rng.normal(size=(count, frames, 17, 2)).astype(np.float32),
             visibility=np.full((count, frames, 17), .8, dtype=np.float32),
             labels=np.arange(count), split=np.array(["train", "train", "train", "probe", "probe", "probe"]),
             frame_counts=np.array([9, 9, 9, 9, 6, 4], dtype=np.int32))
    return PoseData.load(path)


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visibility = []

    def forward(self, x, visibility, frame_mask):
        self.visibility.extend(visibility[b].clone() for b in range(len(x)))
        return {"embedding": torch.zeros(len(x), 2)}


def test_regenerated_masks_match_the_evaluator_bit_for_bit(tmp_path):
    missing = _script("analyze_actual_missingness.py")
    data = _casia_like(tmp_path / "data.npz")
    probes = data.indices(["probe"])
    evaluation = {"corruption_seed": 1729, "corruption_families": ["random_joint", "temporal", "body_part"],
                  "severities": [0.0, .5, .7], "corruption_realizations": 2, "clip_length": None}
    fractions = missing["removed_fractions"](data, probes, evaluation)
    for family_index, family in enumerate(evaluation["corruption_families"]):
        for severity in (.5, .7):
            expected = np.zeros(len(probes))
            for realization in range(2):
                recorder = _Recorder()
                _embeddings(recorder, data, probes, torch.device("cpu"), family, severity,
                            1729 + family_index * 10_000 + realization, {"weight_coordinates": False},
                            batch_size=2, descriptor_builder=lambda output: {"global": output["embedding"]})
                for position, visibility in enumerate(recorder.visibility):
                    length = int(data.frame_counts[probes[position]])
                    expected[position] += float((visibility[:length] == 0).float().mean()) / 2
            assert fractions[(family, severity)] == pytest.approx(expected)
    assert fractions[("body_part", .7)] == pytest.approx(12 / 17)


def _gait3d_release(tmp_path):
    import importlib
    helpers = importlib.import_module("test_gait3d")
    return helpers._release(tmp_path)


def test_fastposegait_gait3d_export_matches_project_splits(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tests"))
    source, partition = _gait3d_release(tmp_path)
    project = tmp_path / "project"
    dataset = project / "data/gait3d/processed/dataset.npz"
    convert_gait3d_pose(source, partition, dataset, progress=None)
    shutil.copytree(ROOT / "configs", project / "configs")
    recipe = project / "configs/recipes/gait3d_v2_available.yaml"
    content = yaml.safe_load(recipe.read_text())
    content["validation"] = {"holdout_identities": 2}
    content["dataset"]["protocol"] = "gait3d_official"
    recipe.write_text(yaml.safe_dump(content))
    for relative in ("gpgait/gpgait_Gait3D.yaml", "gaitgraph2/gaitgraph2_Gait3D.yaml", "default.yaml"):
        target = project / "external/FastPoseGait/configs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "external/FastPoseGait/configs" / relative, target)
    monkeypatch.delenv("OAP_DATA_ROOT", raising=False)
    data = PoseData.load(dataset)

    validation = prepare_fastposegait(project, tmp_path / "val", dataset="gait3d", stage="validation")
    assert validation["official_test_sequences_exported"] == 0 and validation["probes"] == 2
    held = validation["validation_identities"]
    partition_val = json.loads((tmp_path / "val/datasets/Gait3D/Gait3D.json").read_text())
    assert partition_val == json.loads((tmp_path / "val/partition.json").read_text())
    assert {int(i) for i in partition_val["TEST_SET"]} == set(held)
    assert {s.split("-")[0] for s in partition_val["PROBE_SET"]} == set(partition_val["TEST_SET"])
    assert (tmp_path / "val/configs/default.yaml").exists()

    final = prepare_fastposegait(project, tmp_path / "final", dataset="gait3d", stage="final")
    partition_final = json.loads((tmp_path / "final/partition.json").read_text())
    assert partition_final["TRAIN_SET"] == ["0000", "0001", "0002", "0003"]
    assert partition_final["TEST_SET"] == ["0004", "0005"]
    assert partition_final["PROBE_SET"] == [str(s) for s in sorted(data.sequence_ids[data.split == "probe"])]
    assert final["official_test_sequences_exported"] == 4
    emitted = np.load(tmp_path / "final/poses/0000/camid0_videoid1/seq0/seq0.pkl", allow_pickle=True)
    index = int(np.flatnonzero(data.sequence_ids == "0000-camid0_videoid1-seq0")[0])
    assert emitted.shape == (3, 17, 3)
    assert np.array_equal(emitted[..., :2], data.x[index, :3])
    gpgait = yaml.safe_load((tmp_path / "final/gpgait.yaml").read_text())
    gaitgraph2 = yaml.safe_load((tmp_path / "final/gaitgraph2.yaml").read_text())
    assert gpgait["model_cfg"]["num_class"] == 4 and gaitgraph2["model_cfg"]["num_class"] == 128
    assert gpgait["trainer_cfg"]["with_test"] is False and gaitgraph2["trainer_cfg"]["with_test"] is False
    assert gpgait["evaluator_cfg"]["sampler"]["batch_size"] == 1
    assert gaitgraph2["evaluator_cfg"]["sampler"]["batch_size"] == 256
    with pytest.raises(ValueError, match="select distinct"):
        prepare_fastposegait(project, tmp_path / "bad", models=("gpgait++",), dataset="gait3d")


def test_validation_only_recipe_refuses_the_test_stage(monkeypatch, capsys):
    from oap_supcon import cli
    monkeypatch.setattr(sys, "argv", ["cli", "run", "--dataset", "casia_b_pose", "--preset", "casia_validation_v2",
                                      "--method", "ce", "--seed", "11", "--final", "--epochs", "1"])
    with pytest.raises(SystemExit, match="validation-only"):
        cli.main()
