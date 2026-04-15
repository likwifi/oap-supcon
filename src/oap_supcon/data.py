from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
            x = z["x"].astype(np.float32)
            visibility = z["visibility"].astype(np.float32)
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
        if np.any((self.visibility < 0) | (self.visibility > 1)):
            raise ValueError("visibility values must be in [0,1]")
        allowed = {"train", "val", "gallery", "probe", "test"}
        unknown = set(np.unique(self.split)).difference(allowed)
        if unknown:
            raise ValueError(f"unknown split names: {sorted(unknown)}")
        train_ids = set(self.labels[self.split == "train"].tolist())
        test_ids = set(self.labels[np.isin(self.split, ["gallery", "probe", "test"])].tolist())
        overlap = train_ids.intersection(test_ids)
        if overlap:
            raise ValueError(f"identity leakage between train and test: {sorted(overlap)[:10]}")

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for array in (
            self.x,
            self.visibility,
            self.labels,
            self.split,
            self.sequence_ids,
            self.frame_counts,
        ):
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


def normalize_pose(x: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    """Pelvis/torso-center every frame and scale each sequence."""
    weights = visibility.unsqueeze(-1)
    visible_count = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
    visible_center = (x * weights).sum(dim=2, keepdim=True) / visible_count
    roots = _root_indices(x.shape[2])
    root_weights = visibility[:, :, roots].unsqueeze(-1)
    root_count = root_weights.sum(dim=2, keepdim=True)
    root_center = (x[:, :, roots] * root_weights).sum(dim=2, keepdim=True) / root_count.clamp_min(1.0)
    # A missing pelvis falls back to the visible-joint centroid for that frame.
    center = torch.where(root_count > 0, root_center, visible_center)
    centered = (x - center) * weights
    scale = centered.square().sum(dim=-1).sqrt()
    scale = (scale * visibility).sum(dim=(1, 2), keepdim=True) / visibility.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return centered / scale.unsqueeze(-1).clamp_min(1e-4)


class PoseDataset(Dataset):
    def __init__(self, data: PoseData, indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        x = torch.from_numpy(self.data.x[index])
        visibility = torch.from_numpy(self.data.visibility[index])
        x = normalize_pose(x.unsqueeze(0), visibility.unsqueeze(0)).squeeze(0)
        return x, visibility, int(self.data.labels[index]), index


class IdentityBatchSampler(Sampler[list[int]]):
    """P identities x K sequences, ensuring positives exist in every batch."""
    def __init__(self, labels: np.ndarray, identities_per_batch: int, samples_per_identity: int, seed: int):
        self.labels = np.asarray(labels)
        self.p = min(identities_per_batch, len(np.unique(self.labels)))
        self.k = samples_per_identity
        self.seed = seed
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
                batch.extend(rng.choice(pool, self.k, replace=len(pool) < self.k).tolist())
            yield batch


def resolve_dataset_path(project_root: Path, dataset_cfg: dict) -> Path:
    import os

    data_root = Path(os.environ.get("OAP_DATA_ROOT", project_root / "data"))
    return data_root / dataset_cfg["folder"] / dataset_cfg["file"]
