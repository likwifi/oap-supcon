import pickle
from pathlib import Path

import numpy as np
import pytest

from oap_supcon.casia_b import (
    CasiaBSequence,
    convert_casia_b_hrnet,
    load_pose_pickle,
)


def _write_sequence(root: Path, identity: str, sequence: str, view: str, frames: int):
    directory = root / identity / sequence / view
    directory.mkdir(parents=True)
    pose = np.zeros((frames, 17, 3), dtype=np.float32)
    pose[..., 0] = 10
    pose[..., 1] = 20
    pose[..., 2] = 0.75
    with (directory / f"{view}.pkl").open("wb") as handle:
        pickle.dump(pose, handle, protocol=4)
    sequence_id = f"{identity}-{sequence}-{view}"
    mapping = "\n".join(f"{sequence_id}-{frame:03d}" for frame in range(frames))
    (directory / f"{sequence_id}-mapping.txt").write_text(mapping + "\n")


def test_protocol_splits():
    base = Path("/unused")
    train = CasiaBSequence(base, base, "074", "cl-02", "180", 1)
    gallery = CasiaBSequence(base, base, "075", "nm-04", "090", 1)
    probe = CasiaBSequence(base, base, "124", "bg-01", "000", 1)
    assert (train.split, gallery.split, probe.split) == ("train", "gallery", "probe")


def test_converter_preserves_confidence_and_pads(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "dataset.npz"
    _write_sequence(source, "001", "nm-01", "000", 2)
    _write_sequence(source, "075", "nm-01", "018", 3)
    _write_sequence(source, "075", "bg-01", "036", 1)

    data = convert_casia_b_hrnet(
        source, destination, strict_release=False, progress=None
    )

    assert data.x.shape == (3, 3, 17, 2)
    assert data.split.tolist() == ["train", "probe", "gallery"]
    assert np.all(data.visibility[0, :2] == 0.75)
    assert np.all(data.visibility[0, 2] == 0)
    with np.load(destination, allow_pickle=False) as saved:
        assert saved["frame_counts"].tolist() == [2, 1, 3]
        assert saved["pose_source"].item().startswith("CASIA-B_HRNet")


def test_converter_clips_small_hrnet_score_overshoot(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "dataset.npz"
    _write_sequence(source, "001", "nm-01", "000", 1)
    source_path = source / "001" / "nm-01" / "000" / "000.pkl"
    pose = load_pose_pickle(source_path)
    pose[..., 2] = 1.01
    with source_path.open("wb") as handle:
        pickle.dump(pose, handle, protocol=4)

    data = convert_casia_b_hrnet(
        source, destination, strict_release=False, progress=None
    )

    assert np.all(data.visibility == 1.0)


def test_restricted_pickle_loader_rejects_globals(tmp_path):
    path = tmp_path / "unsafe.pkl"
    path.write_bytes(pickle.dumps(eval))
    with pytest.raises(pickle.UnpicklingError, match="forbidden pickle global"):
        load_pose_pickle(path)
