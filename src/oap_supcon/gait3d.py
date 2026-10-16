"""Convert released Gait3D HRNet text poses using an explicit partition."""
from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .casia_b import _write_npz_atomic
from .data import PoseData


_IDENTITY = re.compile(r"\d{4}")
_CAMERA_VIDEO = re.compile(r"(camid\d+)_videoid\d+")
_SEQUENCE = re.compile(r"seq\d+")
_FRAME = re.compile(r"human_crop_f(\d+)\.txt")


def load_partition(path: Path) -> dict[str, list[str]]:
    partition = json.loads(path.read_text(encoding="utf-8"))
    for key in ("TRAIN_SET", "TEST_SET", "PROBE_SET"):
        values = partition.get(key)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError(f"partition {key} must be a list of strings")
        if len(values) != len(set(values)):
            raise ValueError(f"partition {key} contains duplicates")
    train, test = set(partition["TRAIN_SET"]), set(partition["TEST_SET"])
    if not train or not test or train & test:
        raise ValueError("partition needs nonempty, disjoint train and test identities")
    if any(not _IDENTITY.fullmatch(v) for v in train | test):
        raise ValueError("partition identities must have four digits")
    probe_identities = []
    for sequence_id in partition["PROBE_SET"]:
        fields = sequence_id.split("-")
        if (len(fields) != 3 or fields[0] not in test
                or not _CAMERA_VIDEO.fullmatch(fields[1])
                or not _SEQUENCE.fullmatch(fields[2])):
            raise ValueError(f"invalid partition probe: {sequence_id}")
        probe_identities.append(fields[0])
    if Counter(probe_identities) != Counter({identity: 1 for identity in test}):
        raise ValueError("partition must designate exactly one probe per test identity")
    return partition


@dataclass(frozen=True)
class Gait3DSequence:
    path: Path
    identity: str
    camera_video: str
    sequence: str
    frames: int
    frame_gaps: int
    split: str

    @property
    def sequence_id(self) -> str:
        return f"{self.identity}-{self.camera_video}-{self.sequence}"

    @property
    def camera(self) -> str:
        return _CAMERA_VIDEO.fullmatch(self.camera_video).group(1)


def frame_paths(directory: Path) -> list[Path]:
    frames = []
    for entry in os.scandir(directory):
        if entry.name.startswith(".") or not entry.is_file():
            continue
        match = _FRAME.fullmatch(entry.name)
        if match is None:
            raise ValueError(f"unexpected pose frame: {entry.path}")
        frames.append((int(match.group(1)), Path(entry.path)))
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise ValueError(f"empty pose sequence: {directory}")
    if len({number for number, _ in frames}) != len(frames):
        raise ValueError(f"duplicate frame numbers: {directory}")
    return [path for _, path in frames]


def discover_sequences(source: Path, partition: dict) -> tuple[list[Gait3DSequence], dict]:
    if not source.is_dir():
        raise FileNotFoundError(f"Gait3D pose directory not found: {source}")
    train, test, probes = (set(partition[key]) for key in ("TRAIN_SET", "TEST_SET", "PROBE_SET"))
    records = []
    for identity in sorted(source.iterdir()):
        if identity.name.startswith("."):
            continue
        if not identity.is_dir() or identity.name not in train | test:
            raise ValueError(f"identity outside the supplied partition: {identity}")
        for camera in sorted(identity.iterdir()):
            if camera.name.startswith("."):
                continue
            if not camera.is_dir() or not _CAMERA_VIDEO.fullmatch(camera.name):
                raise ValueError(f"unexpected camera/video directory: {camera}")
            for sequence in sorted(camera.iterdir()):
                if sequence.name.startswith("."):
                    continue
                if not sequence.is_dir() or not _SEQUENCE.fullmatch(sequence.name):
                    raise ValueError(f"unexpected sequence directory: {sequence}")
                paths = frame_paths(sequence)
                first, last = (int(_FRAME.fullmatch(p.name).group(1)) for p in (paths[0], paths[-1]))
                key = f"{identity.name}-{camera.name}-{sequence.name}"
                role = "train" if identity.name in train else "probe" if key in probes else "gallery"
                records.append(Gait3DSequence(sequence, identity.name, camera.name,
                                              sequence.name, len(paths), last - first + 1 - len(paths), role))
    if not records:
        raise ValueError(f"no pose sequences under {source}")
    identities = {r.identity for r in records}
    sequences = {r.sequence_id for r in records}
    gallery_ids = {r.identity for r in records if r.split == "gallery"}
    unusable = sorted(r.sequence_id for r in records if r.split == "probe" and r.identity not in gallery_ids)
    report = {
        "expected_train_identities": len(train), "expected_test_identities": len(test),
        "source_identities": len(identities), "source_sequences": len(records),
        "source_frames": sum(r.frames for r in records),
        "missing_train_identities": sorted(train - identities),
        "missing_test_identities": sorted(test - identities),
        "missing_probes": sorted(probes - sequences),
        "probes_without_gallery": unusable,
        "frame_gaps": sum(r.frame_gaps for r in records),
    }
    report["complete"] = not any(report[key] for key in (
        "missing_train_identities", "missing_test_identities", "missing_probes", "probes_without_gallery"))
    return records, report


def load_pose_frame(path: Path) -> np.ndarray:
    # fromstring is much faster than loadtxt for millions of single-row files.
    # Treat partial parses as errors instead of accepting a truncated row.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        try:
            values = np.fromstring(path.read_text(encoding="utf-8"), sep=",", dtype=np.float32)
        except (ValueError, DeprecationWarning) as error:
            raise ValueError(f"{path}: malformed pose row") from error
    if values.shape != (53,) or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected 53 finite values, received {values.shape}")
    if np.any(values[:2] <= 0):
        raise ValueError(f"{path}: crop dimensions must be positive")
    pose = values[2:].reshape(17, 3)
    if np.any(pose[:, 2] < 0):
        raise ValueError(f"{path}: confidence must be nonnegative")
    return pose


def dataset_coverage(path: Path, expected_protocol: str = "gait3d_official") -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if "coverage_json" not in archive:
            raise ValueError("Gait3D dataset needs converter coverage metadata")
        coverage = json.loads(str(archive["coverage_json"].item()))
    if coverage["protocol"] != expected_protocol:
        raise ValueError(f"Gait3D dataset protocol {coverage['protocol']} does not match "
                         f"configured protocol {expected_protocol}; select the matching recipe")
    return {"protocol": coverage["protocol"], "partition_sha256": coverage["partition_sha256"],
            "complete": coverage["complete"], "split_identities": coverage["split_identities"],
            "missing_train_identities": len(coverage["missing_train_identities"]),
            "missing_test_identities": len(coverage["missing_test_identities"]),
            "missing_probes": len(coverage["missing_probes"]),
            "probes_without_gallery": len(coverage["probes_without_gallery"])}


def convert_gait3d_pose(
    source: Path, partition_path: Path, destination: Path, *,
    allow_incomplete: bool = False, report_path: Path | None = None,
    audit_only: bool = False, progress: Callable[[str], None] | None = print,
) -> dict:
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    partition_path = partition_path.expanduser().resolve()
    if destination.exists() and not audit_only:
        raise FileExistsError(f"destination already exists: {destination}")
    if report_path is not None and report_path.expanduser().resolve() == destination:
        raise ValueError("report path must differ from dataset destination")
    partition = load_partition(partition_path)
    records, report = discover_sequences(source, partition)
    report.update({"source": str(source), "partition_sha256": hashlib.sha256(partition_path.read_bytes()).hexdigest()})
    excluded = set(report["probes_without_gallery"])
    retained = [r for r in records if r.sequence_id not in excluded]
    report["excluded_sequences"] = sorted(excluded)
    report["split_sequences"] = dict(Counter(r.split for r in retained))
    report["split_identities"] = {
        role: len({r.identity for r in retained if r.split == role}) for role in ("train", "gallery", "probe")
    }
    report["protocol"] = "gait3d_official" if report["complete"] else "gait3d_available_pose"
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if progress:
        progress(f"coverage: {report['source_identities']} identities; splits={report['split_sequences']}; "
                 f"missing probes={len(report['missing_probes'])}; unscorable probes={len(excluded)}")
    if audit_only:
        return report
    if not report["complete"] and not allow_incomplete:
        raise ValueError("pose release does not cover the full partition; inspect the coverage report, "
                         "then use --allow-incomplete and the gait3d_v2_available recipe for subset experiments")
    if not all(report["split_sequences"].get(role, 0) for role in ("train", "gallery", "probe")):
        raise ValueError("conversion requires nonempty training, gallery and scorable probe splits")
    maximum = max(r.frames for r in retained)
    x = np.zeros((len(retained), maximum, 17, 2), dtype=np.float32)
    visibility = np.zeros(x.shape[:3], dtype=np.float32)
    overshoots = 0
    for index, record in enumerate(retained):
        paths = frame_paths(record.path)
        if len(paths) != record.frames:
            raise ValueError(f"source changed during conversion: {record.path}")
        for frame, path in enumerate(paths):
            pose = load_pose_frame(path)
            x[index, frame] = pose[:, :2]
            overshoots += int((pose[:, 2] > 1).sum())
            visibility[index, frame] = np.clip(pose[:, 2], 0, 1)
        if progress and ((index + 1) % 500 == 0 or index + 1 == len(retained)):
            progress(f"packed {index + 1}/{len(retained)} sequences")
    arrays = {
        "x": x, "visibility": visibility,
        "labels": np.asarray([int(r.identity) for r in retained], dtype=np.int64),
        "split": np.asarray([r.split for r in retained]),
        "sequence_ids": np.asarray([r.sequence_id for r in retained]),
        "conditions": np.full(len(retained), "in_the_wild"),
        "views": np.asarray([r.camera for r in retained]),
        "frame_counts": np.asarray([r.frames for r in retained], dtype=np.int32),
    }
    data = PoseData(**arrays)
    data.validate()
    report.update({"shape": list(x.shape), "confidence_scores_clipped": overshoots,
                   "dataset_sha256": data.checksum()})
    _write_npz_atomic(destination, **arrays,
                      pose_source=np.asarray("Gait3D released HRNet COCO-17 2D pose"),
                      protocol=np.asarray(report["protocol"]),
                      partition_sha256=np.asarray(report["partition_sha256"]),
                      coverage_json=np.asarray(json.dumps(report, sort_keys=True)),
                      confidence_transform=np.asarray("clip HRNet score to [0,1]"))
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if progress:
        progress(f"wrote {destination}; shape={x.shape}; checksum={report['dataset_sha256']}")
    return report
