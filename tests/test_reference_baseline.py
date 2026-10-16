import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from oap_supcon.casia_b import load_pose_pickle
from oap_supcon.data import PoseData
from oap_supcon.reference_baseline import (prepare_fastposegait, reference_configuration,
                                          reference_entries)

ROOT = Path(__file__).resolve().parents[1]


def canonical_fixture(root):
    records = [(identity, sequence, view) for identity in range(1, 125)
               for sequence in ("nm-01", "nm-05") for view in ("000", "090")]
    rng = np.random.default_rng(11)
    x = rng.normal(size=(len(records), 7, 17, 2)).astype(np.float32)
    v = np.full(x.shape[:3], .73, dtype=np.float32)
    lengths = np.full(len(records), 5, np.int32)
    x[:, 5:] = 999; v[:, 5:] = 0
    labels = np.array([identity for identity, _, _ in records])
    split = np.array(["train" if identity < 75 else "gallery" if seq == "nm-01" else "probe"
                      for identity, seq, _ in records])
    path = root / "data/casia_b_pose/processed/dataset.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, x=x, visibility=v, frame_counts=lengths, labels=labels, split=split,
                        sequence_ids=np.array([f"{i:03d}-{seq}-{view}" for i, seq, view in records]),
                        conditions=np.full(len(records), "NM"), views=np.array([view for _, _, view in records]))
    shutil.copytree(ROOT / "configs", root / "configs")
    for model in ["gpgait", "gpgait++", "gaitgraph2"]:
        source = ROOT / "external/FastPoseGait/configs" / model / f"{model}.yaml"
        target = root / "external/FastPoseGait/configs" / model / f"{model}.yaml"
        target.parent.mkdir(parents=True); shutil.copy2(source, target)
    shutil.copy2(ROOT / "external/FastPoseGait/configs/default.yaml",
                 root / "external/FastPoseGait/configs/default.yaml")
    return path


def test_reference_preparation_preserves_raw_pose_and_excludes_official_test(tmp_path, monkeypatch):
    monkeypatch.delenv("OAP_DATA_ROOT", raising=False)
    path = canonical_fixture(tmp_path)
    destination = tmp_path / "reference"
    manifest = prepare_fastposegait(tmp_path, destination)
    assert manifest["prepared"] and manifest["test_sequences_exported"] == 0
    assert len(manifest["training_identities"]) == 62 and len(manifest["validation_identities"]) == 12
    assert manifest["sequences"] == 74 * 4
    assert sorted(p.name for p in (destination / "poses").iterdir()) == [f"{i:03d}" for i in range(1, 75)]
    data = PoseData.load(path)
    emitted = load_pose_pickle(destination / "poses/001/nm-01/000/000.pkl")
    assert emitted.shape == (5, 17, 3)
    assert np.array_equal(emitted[..., :2], data.x[0, :5])
    assert np.array_equal(emitted[..., 2], data.visibility[0, :5])
    partition = json.loads((destination / "partition.json").read_text())
    assert not set(partition["TRAIN_SET"]) & set(partition["TEST_SET"])
    assert {int(i) for i in partition["TEST_SET"]} == set(manifest["validation_identities"])
    for model in ["gpgait", "gpgait++"]:
        cfg = yaml.safe_load((destination / f"{model}.yaml").read_text())
        assert cfg["model_cfg"]["num_class"] == 62
        assert cfg["trainer_cfg"]["with_test"] is False
        assert cfg["evaluator_cfg"]["sampler"]["batch_size"] == 1
        original = yaml.safe_load((tmp_path / "external/FastPoseGait/configs" / model / f"{model}.yaml").read_text())
        for field in ["loss_cfg", "optimizer_cfg", "scheduler_cfg"]:
            assert cfg[field] == original[field]
        assert cfg["trainer_cfg"]["transform"] == original["trainer_cfg"]["transform"]
    with pytest.raises(FileExistsError, match="new preparation"):
        prepare_fastposegait(tmp_path, destination)


def test_reference_dry_run_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.delenv("OAP_DATA_ROOT", raising=False)
    canonical_fixture(tmp_path)
    output = tmp_path / "preview"
    manifest = prepare_fastposegait(tmp_path, output, dry_run=True)
    assert not manifest["prepared"] and not output.exists()
    assert manifest["source_dataset_sha256"] != manifest["development_dataset_sha256"]


def test_reference_rejects_ambiguous_metadata_and_degenerate_clean_frames(tmp_path):
    data = PoseData.load(canonical_fixture(tmp_path))
    wrong = copy.deepcopy(data); wrong.views[0] = "090"
    with pytest.raises(ValueError, match="metadata disagree"):
        reference_entries(wrong, [0])
    with pytest.raises(ValueError, match="duplicate"):
        reference_entries(data, [0, 0])
    data.x[0, 0, :, 1] = 0
    with pytest.raises(ValueError, match="zero-height"):
        reference_entries(data, [0])


def test_reference_configuration_does_not_mutate_the_official_template():
    original = yaml.safe_load((ROOT / "external/FastPoseGait/configs/gpgait/gpgait.yaml").read_text())
    snapshot = copy.deepcopy(original)
    reference_configuration(original, Path("/tmp/example"), [1, 2], 1, "development")
    assert original == snapshot
    with pytest.raises(ValueError, match="positive"):
        reference_configuration(original, Path("/tmp/example"), [1, 2], 0, "development")


def test_casia_final_stage_exports_the_official_protocol_for_reproduction(tmp_path, monkeypatch):
    monkeypatch.delenv("OAP_DATA_ROOT", raising=False)
    canonical_fixture(tmp_path)
    destination = tmp_path / "final"
    manifest = prepare_fastposegait(tmp_path, destination, ("gpgait", "gaitgraph2"), stage="final")
    partition = json.loads((destination / "partition.json").read_text())
    assert partition["TRAIN_SET"] == [f"{i:03d}" for i in range(1, 75)]
    assert partition["TEST_SET"] == [f"{i:03d}" for i in range(75, 125)]
    assert "PROBE_SET" not in partition and manifest["validation_identities"] == []
    assert manifest["test_sequences_exported"] == 50 * 4 and manifest["sequences"] == 124 * 4
    assert "reproduction" in manifest["purpose"]
    gpgait = yaml.safe_load((destination / "gpgait.yaml").read_text())
    gaitgraph2 = yaml.safe_load((destination / "gaitgraph2.yaml").read_text())
    assert gpgait["model_cfg"]["num_class"] == 74 and gaitgraph2["model_cfg"]["num_class"] == 128
    assert (destination / "datasets/CASIA-B/CASIA-B.json").exists()
