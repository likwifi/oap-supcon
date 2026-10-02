from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


REQUIRED_KEYS = {"x", "visibility", "labels", "split"}


@dataclass
class PoseData:
    x: np.ndarray
    visibility: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    sequence_ids: np.ndarray
    conditions: np.ndarray
    views: np.ndarray
    frame_counts: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path) -> "PoseData":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}. See data/README.md and set OAP_DATA_ROOT "
                "when data are stored outside this project."
            )
        with np.load(path, allow_pickle=False) as z:
            missing = REQUIRED_KEYS.difference(z.files)
            if missing:
                raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
            x = z["x"].astype(np.float32, copy=False)
            visibility = z["visibility"].astype(np.float32, copy=False)
            labels = z["labels"].astype(np.int64)
            split = z["split"].astype(str)
            n = len(labels)
            sequence_ids = z["sequence_ids"].astype(str) if "sequence_ids" in z else np.arange(n).astype(str)
            conditions = z["conditions"].astype(str) if "conditions" in z else np.full(n, "unknown")
            views = z["views"].astype(str) if "views" in z else np.full(n, "unknown")
            frame_counts = (
                z["frame_counts"].astype(np.int32)
                if "frame_counts" in z
                else np.full(n, x.shape[1], dtype=np.int32)
            )
        data = cls(x, visibility, labels, split, sequence_ids, conditions, views, frame_counts)
        data.validate()
        return data

    def validate(self) -> None:
        if self.x.ndim != 4:
            raise ValueError(f"x must have shape [N,T,J,D], received {self.x.shape}")
        if self.visibility.shape != self.x.shape[:3]:
            raise ValueError("visibility must have shape [N,T,J]")
        n = self.x.shape[0]
        for name in ("labels", "split", "sequence_ids", "conditions", "views"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length does not match x")
        if self.frame_counts is None:
            self.frame_counts = np.full(n, self.x.shape[1], dtype=np.int32)
        if self.frame_counts.shape != (n,):
            raise ValueError("frame_counts must have shape [N]")
        if np.any((self.frame_counts < 1) | (self.frame_counts > self.x.shape[1])):
            raise ValueError("frame_counts values must be between 1 and padded T")
        if self.x.shape[-1] not in (2, 3):
            raise ValueError("coordinate dimension D must be 2 or 3")
        if not np.isfinite(self.x).all():
            raise ValueError("x contains NaN or infinity")
        if not np.isfinite(self.visibility).all() or np.any((self.visibility < 0) | (self.visibility > 1)):
            raise ValueError("visibility values must be in [0,1]")
        allowed = {"train", "val", "val_gallery", "val_probe", "gallery", "probe", "test"}
        unknown = set(np.unique(self.split)).difference(allowed)
        if unknown:
            raise ValueError(f"unknown split names: {sorted(unknown)}")
        train_ids = set(self.labels[self.split == "train"].tolist())
        test_ids = set(self.labels[np.isin(self.split, ["gallery", "probe", "test"])].tolist())
        overlap = train_ids.intersection(test_ids)
        if overlap:
            raise ValueError(f"identity leakage between train and test: {sorted(overlap)[:10]}")
        val_ids = set(self.labels[np.isin(self.split, ["val", "val_gallery", "val_probe"])].tolist())
        if val_ids & (train_ids | test_ids):
            raise ValueError("identity leakage involving validation identities")

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for array in (
            self.x,
            self.visibility,
            self.labels,
            self.split,
            self.sequence_ids,
            self.frame_counts,
            self.conditions,
            self.views,
        ):
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def indices(self, split_names: Iterable[str]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.split, list(split_names)))


def _root_indices(joints: int) -> list[int]:
    if joints == 17:  # COCO pelvis midpoint
        return [11, 12]
    if joints == 18:  # OpenPose hip midpoint
        return [8, 11]
    if joints == 25:  # Kinect spine base
        return [0]
    return list(range(joints))


def normalize_pose(
    x: torch.Tensor, visibility: torch.Tensor, weight_coordinates: bool = True
) -> torch.Tensor:
    """Pelvis/torso-center every frame and scale each sequence.

    `weight_coordinates` reproduces the original behaviour, where every joint is
    multiplied by its visibility. For a dataset whose visibility channel is a
    continuous detector confidence -- CASIA-B HRNet is 0.852 +- 0.116 and never
    exactly zero -- that is not a mask but a radial shrink of 14.8% +- 11.6% per
    joint, against a between-sequence anatomical spread of only ~29%. Confidence
    already reaches the encoder as an input channel and weights the masked-mean
    pooling, so it should not also deform the geometry; set this to False.
    Genuinely masked joints are still zeroed, because `corrupt` multiplies the
    coordinates by the mask after normalization.
    """
    weights = visibility.unsqueeze(-1)
    visible_count = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
    visible_center = (x * weights).sum(dim=2, keepdim=True) / visible_count
    roots = _root_indices(x.shape[2])
    root_weights = visibility[:, :, roots].unsqueeze(-1)
    root_count = root_weights.sum(dim=2, keepdim=True)
    root_center = (x[:, :, roots] * root_weights).sum(dim=2, keepdim=True) / root_count.clamp_min(1e-6)
    # A missing pelvis falls back to the visible-joint centroid for that frame.
    center = torch.where(root_count > 0, root_center, visible_center)
    centered = x - center
    if weight_coordinates:
        centered = centered * weights
    else:
        # Keep padding and fully absent joints at the origin without rescaling
        # the joints that are present.
        centered = centered * (weights > 0).to(centered.dtype)
    scale = centered.square().sum(dim=-1).sqrt()
    scale = (scale * visibility).sum(dim=(1, 2), keepdim=True) / visibility.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return centered / scale.unsqueeze(-1).clamp_min(1e-4)


class PoseDataset(Dataset):
    """Sequences, optionally resampled to a fixed-length clip.

    Sequences are stored zero-padded to the longest clip in the dataset. For
    CASIA-B that is T=306 against a mean real length of 96.7, so 68% of every
    tensor is padding. Masked-mean pooling excludes it, but the convolutional
    trunk and its BatchNorms do not: measured inside the temporal trunk, real
    frames have mean activation +4.55 and padded frames +9.63, and the stored
    running mean is +8.03 -- exactly 0.32*4.55 + 0.68*9.63. Every real
    activation is therefore normalised by statistics two thirds determined by
    padding.

    Setting `clip_length` samples a contiguous window of that many real frames
    (wrapping around for shorter sequences) so the batch contains no padding at
    all. This is also what the reference baselines do: GaitGraph2 and GaitTR
    both use `frames_num_fixed: 60`.
    """

    def __init__(
        self,
        data: PoseData,
        indices: np.ndarray,
        weight_coordinates: bool = True,
        clip_length: int | None = None,
        clip_mode: str = "center",
        seed: int = 0,
    ):
        if clip_mode not in {"random", "center"}:
            raise ValueError(f"unknown clip_mode: {clip_mode}")
        self.data = data
        self.indices = np.asarray(indices)
        self.weight_coordinates = bool(weight_coordinates)
        self.clip_length = None if not clip_length else int(clip_length)
        self.clip_mode = clip_mode
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        x = torch.from_numpy(self.data.x[index])
        visibility = torch.from_numpy(self.data.visibility[index])
        length = int(self.data.frame_counts[index])
        if self.clip_length:
            span = max(1, length)
            if self.clip_mode == "random":
                start = int(self._rng.integers(max(1, span - self.clip_length + 1)))
            else:
                start = max(0, (span - self.clip_length) // 2)
            # Wrap around so a clip shorter than the window repeats its own
            # frames rather than being padded back to the buffer length.
            positions = (start + np.arange(self.clip_length)) % span
            x, visibility = x[positions], visibility[positions]
            length = self.clip_length
        x = normalize_pose(
            x.unsqueeze(0), visibility.unsqueeze(0), self.weight_coordinates
        ).squeeze(0)
        return x, visibility, length, int(self.data.labels[index]), index


class IdentityBatchSampler(Sampler[list[int]]):
    """P identities x K sequences, ensuring positives exist in every batch."""
    def __init__(self, labels: np.ndarray, identities_per_batch: int, samples_per_identity: int, seed: int,
                 views: np.ndarray | None = None):
        if identities_per_batch < 1 or samples_per_identity < 1:
            raise ValueError("P and K must be positive")
        self.labels = np.asarray(labels)
        self.p = min(identities_per_batch, len(np.unique(self.labels)))
        self.k = samples_per_identity
        self.seed = seed
        self.views = None if views is None else np.asarray(views)
        self.by_identity = {int(label): np.flatnonzero(self.labels == label) for label in np.unique(self.labels)}
        self.length = max(1, len(self.labels) // max(1, self.p * self.k))

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        identities = np.array(list(self.by_identity))
        for _ in range(self.length):
            chosen_ids = rng.choice(identities, self.p, replace=len(identities) < self.p)
            batch: list[int] = []
            for identity in chosen_ids:
                pool = self.by_identity[int(identity)]
                if self.views is None:
                    chosen = rng.choice(pool, self.k, replace=len(pool) < self.k).tolist()
                else:
                    # Prefer distinct camera views before repeating a view.
                    chosen = []
                    camera_order = rng.permutation(np.unique(self.views[pool]))
                    for camera in camera_order[:self.k]:
                        chosen.append(int(rng.choice(pool[self.views[pool] == camera])))
                    available = np.setdiff1d(pool, chosen)
                    if len(chosen) < self.k:
                        source = available if len(available) else pool
                        chosen.extend(rng.choice(source, self.k - len(chosen),
                                                 replace=len(source) < self.k - len(chosen)).tolist())
                batch.extend(chosen)
            yield batch


def resolve_dataset_path(project_root: Path, dataset_cfg: dict) -> Path:
    import os

    data_root = Path(os.environ.get("OAP_DATA_ROOT", project_root / "data"))
    return data_root / dataset_cfg["folder"] / dataset_cfg["file"]


def development_split(data: PoseData, cfg: dict) -> tuple[PoseData, list[int]]:
    """Create CASIA-B validation roles without copying poses or touching test IDs.

    The holdout seed is a protocol choice and is independent of training seeds.
    Other datasets must supply explicit identity-disjoint val_gallery/val_probe.
    """
    val = cfg.get("validation", {})
    if not val.get("enabled", False):
        return data, []
    if len(data.indices(["val_gallery"])) and len(data.indices(["val_probe"])):
        data.validate()
        return data, sorted(np.unique(data.labels[data.indices(["val_gallery", "val_probe"])]).tolist())
    if cfg["dataset"]["name"] != "casia_b_pose":
        raise ValueError("this dataset needs explicit val_gallery and val_probe splits")
    train_ids = np.unique(data.labels[data.split == "train"])
    count = int(val.get("holdout_identities", 12))
    if not 2 <= count <= len(train_ids) - 2:
        raise ValueError("hold out at least two identities and retain at least two for training")
    held = np.sort(np.random.default_rng(int(val.get("split_seed", 2026))).choice(train_ids, count, replace=False))
    split = data.split.astype("U16", copy=True)
    selected = (split == "train") & np.isin(data.labels, held)
    # Converter sequence IDs are identity-sequence-view, e.g. 001-nm-01-090.
    import re
    gallery = np.array([bool(re.search(r"-nm-0[1-4]-\d{3}$", str(s))) for s in data.sequence_ids])
    split[selected & gallery] = "val_gallery"
    split[selected & ~gallery] = "val_probe"
    for identity in held:
        for role in ("val_gallery", "val_probe"):
            if not np.any((data.labels == identity) & (split == role)):
                raise ValueError(f"validation identity {identity} has no {role}; check sequence IDs")
    result = replace(data, split=split)
    result.validate()
    return result, held.tolist()
