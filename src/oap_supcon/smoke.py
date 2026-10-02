from __future__ import annotations

from pathlib import Path

import numpy as np


def make_smoke_dataset(path: Path, seed: int = 2026):
    """Create identity-disjoint synthetic gait-like trajectories for pipeline testing only."""
    rng = np.random.default_rng(seed)
    frames, joints, dimensions = 24, 17, 2
    records = []
    # Training and evaluation identities are deliberately disjoint.
    for identity in range(12):
        evaluation_identity = identity >= 8
        count = 5 if not evaluation_identity else 4
        phase_bias = identity * 0.13
        body_scale = 0.8 + identity * 0.04
        for sequence in range(count):
            t = np.linspace(0, 2 * np.pi, frames, endpoint=False)
            base_x = np.linspace(-0.5, 0.5, joints)[None, :]
            base_y = np.linspace(0.8, -0.8, joints)[None, :]
            gait = np.sin(t[:, None] * 2 + np.arange(joints)[None, :] * 0.2 + phase_bias)
            x = np.stack([
                body_scale * base_x + 0.12 * gait,
                body_scale * base_y + 0.06 * np.cos(t[:, None] * 2 + phase_bias),
            ], axis=-1).astype(np.float32)
            x += rng.normal(0, 0.01, x.shape).astype(np.float32)
            visibility = np.ones((frames, joints), np.float32)
            if evaluation_identity:
                split = "gallery" if sequence == 0 else "probe"
            elif identity >= 6:
                split = "val_gallery" if sequence == 0 else "val_probe"
            else:
                split = "train"
            records.append((x, visibility, identity, split, f"id{identity:03d}_seq{sequence:02d}", "NM", "090"))
    arrays = list(zip(*records))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=np.stack(arrays[0]), visibility=np.stack(arrays[1]), labels=np.asarray(arrays[2]),
        split=np.asarray(arrays[3]), sequence_ids=np.asarray(arrays[4]),
        conditions=np.asarray(arrays[5]), views=np.asarray(arrays[6]),
    )
    return path
