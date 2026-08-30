from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ArrayLike = np.ndarray


def clip_update(update: ArrayLike, max_norm: float) -> np.ndarray:
    if not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be a finite positive number")
    update = np.asarray(update, dtype=float)
    if update.size == 0:
        raise ValueError("update must not be empty")
    if not np.all(np.isfinite(update)):
        raise ValueError("update contains non-finite values")
    norm = np.linalg.norm(update)
    if norm == 0 or norm <= max_norm:
        return update.copy()
    return update * (max_norm / norm)


def gaussian_noise(
    shape: tuple[int, ...],
    sigma: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be a finite non-negative number")
    if not np.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("sensitivity must be a finite positive number")
    return rng.normal(0.0, sigma * sensitivity, size=shape)


def fedavg(updates: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    if not updates:
        raise ValueError("at least one client update is required")
    arrays = [np.asarray(u, dtype=float) for u in updates]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("all client updates must have the same shape")
    if any(a.size == 0 or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("client updates must be non-empty and finite")
    if weights is None:
        weights = [1.0] * len(arrays)
    if len(weights) != len(arrays):
        raise ValueError("weights must match the number of updates")
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and have a positive sum")
    w /= w.sum()
    return np.sum(np.stack(arrays) * w.reshape((-1,) + (1,) * len(shape)), axis=0)


def secure_aggregate(
    updates: Sequence[np.ndarray],
    max_norm: float,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Clip client updates and optionally add Gaussian noise.

    This is an educational DP-oriented primitive, not a complete DP-FL protocol.
    Formal guarantees require an explicit privacy accountant, sampling model,
    participation assumptions, clipping bound, round count and (epsilon, delta).
    """
    clipped = [clip_update(u, max_norm) for u in updates]
    aggregate = fedavg(clipped)
    if not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("noise_sigma must be a finite non-negative number")
    if noise_sigma == 0:
        return aggregate
    generator = np.random.default_rng(rng)
    return aggregate + gaussian_noise(aggregate.shape, noise_sigma, max_norm, generator)
