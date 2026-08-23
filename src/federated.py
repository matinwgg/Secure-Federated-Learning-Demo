from __future__ import annotations

import numpy as np


def clip_update(update: np.ndarray, max_norm: float) -> np.ndarray:
    update = np.asarray(update, dtype=float)
    norm = np.linalg.norm(update)
    if norm == 0 or norm <= max_norm:
        return update.copy()
    return update * (max_norm / norm)


def gaussian_noise(shape: tuple[int, ...], sigma: float, sensitivity: float, rng: np.random.Generator) -> np.ndarray:
    if sigma < 0 or sensitivity < 0:
        raise ValueError("sigma and sensitivity must be non-negative")
    return rng.normal(0.0, sigma * sensitivity, size=shape)


def fedavg(updates: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if not updates:
        raise ValueError("at least one client update is required")
    arrays = [np.asarray(u, dtype=float) for u in updates]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("all client updates must have the same shape")
    if weights is None:
        weights = [1.0] * len(arrays)
    if len(weights) != len(arrays) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("invalid aggregation weights")
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    return sum(a * wi for a, wi in zip(arrays, w))


def secure_aggregate(updates: list[np.ndarray], max_norm: float, noise_sigma: float = 0.0, seed: int = 0) -> np.ndarray:
    clipped = [clip_update(u, max_norm) for u in updates]
    aggregate = fedavg(clipped)
    rng = np.random.default_rng(seed)
    return aggregate + gaussian_noise(aggregate.shape, noise_sigma, max_norm, rng)
