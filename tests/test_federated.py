import numpy as np
import pytest

from federated import clip_update, fedavg, secure_aggregate


def test_clip_limits_norm():
    out = clip_update(np.array([3.0, 4.0]), 1.0)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_clip_does_not_mutate_input():
    original = np.array([3.0, 4.0])
    clip_update(original, 1.0)
    assert np.array_equal(original, [3.0, 4.0])


def test_weighted_fedavg():
    out = fedavg([np.array([1.0, 3.0]), np.array([3.0, 5.0])], [1, 3])
    assert np.allclose(out, [2.5, 4.5])


def test_fedavg_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        fedavg([np.array([1.0]), np.array([1.0, 2.0])])


def test_fedavg_rejects_invalid_weights():
    with pytest.raises(ValueError):
        fedavg([np.array([1.0]), np.array([2.0])], [1, -1])


def test_clip_rejects_non_finite_values():
    with pytest.raises(ValueError):
        clip_update(np.array([1.0, np.nan]), 1.0)


def test_secure_aggregate_is_reproducible_from_seed():
    updates = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]
    first = secure_aggregate(updates, 10, 0.1, 42)
    second = secure_aggregate(updates, 10, 0.1, 42)
    assert np.allclose(first, second)


def test_secure_aggregate_rejects_invalid_noise():
    with pytest.raises(ValueError):
        secure_aggregate([np.array([1.0])], 1.0, -0.1)
