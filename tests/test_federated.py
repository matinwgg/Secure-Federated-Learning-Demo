import numpy as np
from federated import clip_update, fedavg, secure_aggregate


def test_clip_limits_norm():
    out = clip_update(np.array([3.0, 4.0]), 1.0)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_weighted_fedavg():
    out = fedavg([np.array([1.0, 3.0]), np.array([3.0, 5.0])], [1, 3])
    assert np.allclose(out, [2.5, 4.5])


def test_secure_aggregate_is_reproducible():
    updates = [np.array([1., 2.]), np.array([2., 3.])]
    assert np.allclose(secure_aggregate(updates, 10, .1, 42), secure_aggregate(updates, 10, .1, 42))
