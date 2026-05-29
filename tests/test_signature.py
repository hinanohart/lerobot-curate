"""Property tests for the truncated path signature (critic patch P2).

(a) Chen identity, (b) shuffle product, (c) known-path analytic solution.
"""

import numpy as np
import pytest

from lerobot_curate.core.signature import (
    chen_product,
    lead_lag_transform,
    path_signature,
    segment_signature,
    signature_features,
    signed_area,
    time_augment,
)


def test_known_single_segment_analytic():
    # (c) A single linear segment 0 -> a has signature exp(a):
    #     level1 = a, level2 = a⊗a/2, level3 = a⊗a⊗a/6
    a = np.array([2.0, -1.0, 0.5])
    pts = np.vstack([np.zeros(3), a])
    sig = path_signature(pts, 3)
    assert np.allclose(sig[1], a)
    assert np.allclose(sig[2], np.multiply.outer(a, a) / 2.0)
    assert np.allclose(sig[3], np.multiply.outer(np.multiply.outer(a, a), a) / 6.0)
    # segment_signature must agree with path_signature on one segment
    seg = segment_signature(a, 3)
    for k in range(4):
        assert np.allclose(seg[k], sig[k])


def test_chen_identity():
    # (a) sig(concat(P1, P2)) == chen_product(sig(P1), sig(P2))
    rng = np.random.RandomState(1)
    d, depth = 3, 3
    p1 = np.cumsum(rng.randn(5, d), axis=0)
    p2_inc = np.cumsum(rng.randn(4, d), axis=0)
    p2 = p1[-1] + np.vstack([np.zeros(d), p2_inc])  # starts where p1 ends
    full = np.vstack([p1, p2[1:]])
    s1 = path_signature(p1, depth)
    s2 = path_signature(p2, depth)
    s_full = path_signature(full, depth)
    s_chen = chen_product(s1, s2, depth, d)
    for k in range(depth + 1):
        assert np.allclose(s_full[k], s_chen[k], atol=1e-9), f"Chen identity fails at level {k}"


def test_shuffle_product():
    # (b) shuffle relation for single letters: S^i * S^j = S^{ij} + S^{ji}
    rng = np.random.RandomState(2)
    d = 4
    pts = np.cumsum(rng.randn(7, d), axis=0)
    sig = path_signature(pts, 2)
    lvl1, lvl2 = sig[1], sig[2]
    for i in range(d):
        for j in range(d):
            lhs = lvl1[i] * lvl1[j]
            rhs = lvl2[i, j] + lvl2[j, i]
            assert abs(lhs - rhs) < 1e-9, f"shuffle fails at ({i},{j})"


def test_signed_area_triangle():
    # signed area of triangle (0,0)->(1,0)->(0,1)->(0,0) is +0.5
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert abs(signed_area(tri) - 0.5) < 1e-9
    # reversed orientation flips sign
    assert abs(signed_area(tri[::-1]) + 0.5) < 1e-9


def test_closed_loop_zero_level1():
    # a path returning to its start has zero level-1 signature
    rng = np.random.RandomState(3)
    pts = np.cumsum(rng.randn(6, 3), axis=0)
    closed = np.vstack([pts, pts[0]])
    sig = path_signature(closed, 2)
    assert np.allclose(sig[1], 0.0, atol=1e-9)


def test_transform_shapes():
    pts = np.random.RandomState(4).randn(6, 3)
    ll = lead_lag_transform(pts)
    assert ll.shape == (2 * (6 - 1) + 1, 6)
    ta = time_augment(pts)
    assert ta.shape == (6, 4)
    assert np.allclose(ta[:, 0], np.linspace(0, 1, 6))


def test_signature_features_dim_and_determinism():
    pts = np.random.RandomState(5).randn(8, 2)
    f1 = signature_features(pts, depth=3, use_lead_lag=True, use_time_aug=True)
    f2 = signature_features(pts, depth=3, use_lead_lag=True, use_time_aug=True)
    assert np.array_equal(f1, f2)  # deterministic
    # after lead-lag (2*2=4) + time (4+1=5), depth-3 features = 5 + 25 + 125
    big_d = 2 * 2 + 1
    assert f1.shape[0] == big_d + big_d**2 + big_d**3


def test_single_point_raises_or_degenerates():
    with pytest.raises(ValueError):
        signature_features(np.zeros((0, 3)))
    # a path of one point -> no movement -> zero level-1 features
    sig = path_signature(np.zeros((1, 3)), 3)
    assert np.allclose(sig[1], 0.0)
