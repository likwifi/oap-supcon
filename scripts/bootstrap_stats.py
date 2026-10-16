"""Crossed seed-by-identity bootstrap shared by the run analyzers.

Every training seed is evaluated on the same probe identities, so identity is
crossed with seed rather than nested inside it. One identity draw is therefore
shared by all sampled seeds. Drawing identities separately for each seed
averages identity variation away across seeds and understates the interval.
"""
from __future__ import annotations

import numpy as np


def crossed_draws(values, samples: int, rng: np.random.Generator, chunk: int = 2000) -> np.ndarray:
    """Bootstrap means of a [seed, identity] matrix of paired values."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"expected a non-empty [seed, identity] matrix, received {values.shape}")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    seeds, identities = values.shape
    draws = np.empty(samples)
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        seed_draws = rng.integers(0, seeds, size=(size, seeds))
        identity_draws = rng.integers(0, identities, size=(size, identities))
        draws[start:start + size] = values[seed_draws[:, :, None], identity_draws[:, None, :]].mean(axis=(1, 2))
    return draws


def crossed_interval(values, samples: int, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=np.float64)
    draws = crossed_draws(values, samples, rng)
    return {
        "difference": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [.025, .975])],
        "probability_positive": float(np.mean(draws > 0)),
        "seed_blocks": int(values.shape[0]),
        "identity_blocks": int(values.shape[1]),
        "design": "crossed seed x identity",
    }
