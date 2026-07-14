from __future__ import annotations

import os
import pickle
import re
import tempfile
from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .data import PoseData


TRAIN_IDENTITIES = frozenset(range(1, 75))
TEST_IDENTITIES = frozenset(range(75, 125))
SEQUENCES = tuple(
    [f"nm-{index:02d}" for index in range(1, 7)]
    + [f"bg-{index:02d}" for index in range(1, 3)]
    + [f"cl-{index:02d}" for index in range(1, 3)]
)
VIEWS = tuple(f"{angle:03d}" for angle in range(0, 181, 18))
GALLERY_SEQUENCES = frozenset({"nm-01", "nm-02", "nm-03", "nm-04"})
PROBE_SEQUENCES = frozenset(set(SEQUENCES).difference(GALLERY_SEQUENCES))

# These three source sequences are absent from the ScienceDB HRNet release.
# A future release may add them, but no other missing sequence is accepted by
# the default strict validation.
KNOWN_RELEASE_GAPS = frozenset(
    {
        ("094", "nm-01", "000"),
        ("094", "nm-03", "000"),
        ("109", "nm-01", "144"),
    }
)

_IDENTITY_RE = re.compile(r"\d{3}")
_SEQUENCE_RE = re.compile(r"(?:nm-0[1-6]|bg-0[1-2]|cl-0[1-2])")
_VIEW_RE = re.compile(r"(?:000|018|036|054|072|090|108|126|144|162|180)")


try:
    _numpy_multiarray = import_module("numpy._core.multiarray")
except ModuleNotFoundError:  # NumPy 1.x
    _numpy_multiarray = import_module("numpy.core.multiarray")


class _NumpyArrayUnpickler(pickle.Unpickler):
    """Load only the NumPy globals needed by the official array pickles."""

    _ALLOWED: dict[tuple[str, str], object] = {
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("numpy.core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): _numpy_multiarray.scalar,
        ("numpy._core.multiarray", "scalar"): _numpy_multiarray.scalar,
    }

    def find_class(self, module: str, name: str):
        allowed = self._ALLOWED.get((module, name))
        if allowed is None:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")
        return allowed


def load_pose_pickle(path: Path) -> np.ndarray:
    """Safely load and validate one FastPoseGait CASIA-B pose array."""
    with path.open("rb") as handle:
        value = _NumpyArrayUnpickler(handle).load()
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{path}: expected a NumPy array, received {type(value).__name__}")
    if value.ndim != 3 or value.shape[1:] != (17, 3):
        raise ValueError(f"{path}: expected [T,17,3], received {value.shape}")
    if value.shape[0] == 0:
        raise ValueError(f"{path}: empty pose sequence")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{path}: expected numeric pose data, received {value.dtype}")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: pose data contain NaN or infinity")
    confidence = value[..., 2]
    if np.any(confidence < 0):
        raise ValueError(
            f"{path}: HRNet confidence must be nonnegative, received "
            f"minimum {float(confidence.min())}"
        )
    return value


@dataclass(frozen=True)
class CasiaBSequence:
    path: Path
    mapping_path: Path
    identity: str
    sequence: str
    view: str
    frames: int

    @property
    def sequence_id(self) -> str:
        return f"{self.identity}-{self.sequence}-{self.view}"

    @property
    def condition(self) -> str:
        return self.sequence[:2].upper()

    @property
    def split(self) -> str:
        identity = int(self.identity)
        if identity in TRAIN_IDENTITIES:
            return "train"
        if identity in TEST_IDENTITIES:
            return "gallery" if self.sequence in GALLERY_SEQUENCES else "probe"
        raise ValueError(f"identity outside the CASIA-B 001-124 protocol: {self.identity}")


def _source_fields(path: Path) -> tuple[str, str, str]:
    try:
        identity = path.parents[2].name
        sequence = path.parents[1].name
        view = path.parent.name
    except IndexError as error:
        raise ValueError(f"unexpected source path: {path}") from error
    if not _IDENTITY_RE.fullmatch(identity):
        raise ValueError(f"{path}: expected a three-digit identity directory")
    if not _SEQUENCE_RE.fullmatch(sequence):
        raise ValueError(f"{path}: unexpected CASIA-B sequence directory {sequence!r}")
    if not _VIEW_RE.fullmatch(view):
        raise ValueError(f"{path}: unexpected CASIA-B view directory {view!r}")
    if path.stem != view:
        raise ValueError(f"{path}: pickle filename must match its view directory")
    return identity, sequence, view


def _validate_mapping(path: Path, sequence_id: str, frames: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"mapping file missing for {sequence_id}: {path}")
    with path.open(encoding="utf-8") as handle:
        frame_ids = [line.strip() for line in handle if line.strip()]
    if len(frame_ids) != frames:
        raise ValueError(
            f"{path}: mapping has {len(frame_ids)} frames but the pose array has {frames}"
        )
    expected_prefix = f"{sequence_id}-"
    if any(not frame_id.startswith(expected_prefix) for frame_id in frame_ids):
        raise ValueError(f"{path}: mapping contains a frame from another sequence")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError(f"{path}: mapping contains duplicate frame identifiers")


def discover_sequences(
    source_root: Path,
    *,
    strict_release: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[CasiaBSequence]:
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"CASIA-B HRNet directory not found: {source_root}")
    paths = sorted(source_root.rglob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no pose pickles found under {source_root}")

    records: list[CasiaBSequence] = []
    keys: set[tuple[str, str, str]] = set()
    for index, path in enumerate(paths, start=1):
        identity, sequence, view = _source_fields(path)
        key = (identity, sequence, view)
        if key in keys:
            raise ValueError(f"duplicate CASIA-B sequence: {'-'.join(key)}")
        keys.add(key)
        pose = load_pose_pickle(path)
        sequence_id = f"{identity}-{sequence}-{view}"
        mapping_path = path.with_name(f"{sequence_id}-mapping.txt")
        _validate_mapping(mapping_path, sequence_id, pose.shape[0])
        records.append(
            CasiaBSequence(path, mapping_path, identity, sequence, view, pose.shape[0])
        )
        if progress is not None and (index % 1000 == 0 or index == len(paths)):
            progress(f"validated {index}/{len(paths)} source sequences")

    expected = {
        (f"{identity:03d}", sequence, view)
        for identity in range(1, 125)
        for sequence in SEQUENCES
        for view in VIEWS
    }
    unexpected = keys.difference(expected)
    missing = expected.difference(keys)
    if unexpected:
        raise ValueError(f"unexpected source sequences: {sorted(unexpected)[:10]}")
    if strict_release:
        unexpected_missing = missing.difference(KNOWN_RELEASE_GAPS)
        if unexpected_missing:
            raise ValueError(
                "source is incomplete; unexpected missing sequences include "
                f"{sorted(unexpected_missing)[:10]}"
            )
    return records


def _write_npz_atomic(destination: Path, **arrays: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def convert_casia_b_hrnet(
    source_root: Path,
    destination: Path,
    *,
    strict_release: bool = True,
    progress: Callable[[str], None] | None = print,
) -> PoseData:
    """Convert the ScienceDB/FastPoseGait HRNet release to canonical NPZ."""
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination}; remove or rename it explicitly first"
        )
    records = discover_sequences(
        source_root, strict_release=strict_release, progress=progress
    )
    maximum_frames = max(record.frames for record in records)
    count = len(records)
    x = np.zeros((count, maximum_frames, 17, 2), dtype=np.float32)
    visibility = np.zeros((count, maximum_frames, 17), dtype=np.float32)

    for index, record in enumerate(records, start=1):
        pose = load_pose_pickle(record.path)
        x[index - 1, : record.frames] = pose[..., :2]
        # The official HRNet pickles contain occasional small heatmap-score
        # overshoots above 1.0. Canonical visibility is explicitly [0,1].
        visibility[index - 1, : record.frames] = np.clip(pose[..., 2], 0.0, 1.0)
        if progress is not None and (index % 1000 == 0 or index == count):
            progress(f"packed {index}/{count} source sequences")

    labels = np.asarray([int(record.identity) for record in records], dtype=np.int64)
    split = np.asarray([record.split for record in records])
    sequence_ids = np.asarray([record.sequence_id for record in records])
    conditions = np.asarray([record.condition for record in records])
    views = np.asarray([record.view for record in records])
    frame_counts = np.asarray([record.frames for record in records], dtype=np.int32)

    data = PoseData(
        x, visibility, labels, split, sequence_ids, conditions, views, frame_counts
    )
    data.validate()
    _write_npz_atomic(
        destination,
        x=x,
        visibility=visibility,
        labels=labels,
        split=split,
        sequence_ids=sequence_ids,
        conditions=conditions,
        views=views,
        frame_counts=frame_counts,
        pose_source=np.asarray("CASIA-B_HRNet (FastPoseGait/ScienceDB)"),
        protocol=np.asarray("CASIA-B 74/50; nm-01..04 gallery; remaining sequences probe"),
        confidence_transform=np.asarray("clip HRNet score to [0,1]"),
    )
    if progress is not None:
        progress(f"wrote {destination}")
    return data
