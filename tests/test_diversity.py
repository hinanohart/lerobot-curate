"""Tests for diversity scoring and budget subset selection.

Includes a golden test against the reference ``vendi-score`` package and a
monotonicity test (redundancy must reduce diversity).
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from lerobot_curate.core.diversity import (
    greedy_dpp_map,
    normalized_gram,
    rff_features,
    select_diverse,
    vendi_score,
    von_neumann_entropy,
)

vendi_pkg = pytest.importorskip("vendi_score.vendi")


def test_vendi_matches_reference_package():
    rng = np.random.RandomState(0)
    x = rng.randn(12, 6)
    k = np.exp(-cdist(x, x, "sqeuclidean"))  # unit-diagonal RBF gram
    assert abs(vendi_score(k) - float(vendi_pkg.score_K(k, normalize=False))) < 1e-6


def test_vendi_identical_is_one():
    feats = np.tile(np.array([1.0, 2.0, 3.0]), (20, 1))
    k = normalized_gram(rff_features(feats, 512, gamma=0.5, seed=0))
    assert abs(vendi_score(k) - 1.0) < 1e-6


def test_vendi_distinct_approaches_n():
    # well-separated points -> Vendi close to n
    feats = np.eye(10) * 50.0
    v = vendi_score(normalized_gram(rff_features(feats, 1024, gamma=0.01, seed=0)))
    assert v > 8.0  # close to 10


def test_vendi_monotone_decreasing_with_redundancy():
    rng = np.random.RandomState(1)
    base = rng.randn(20, 8) * 5.0
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8]
    vendis = []
    for frac in fractions:
        feats = base.copy()
        n_dup = int(frac * 20)
        # overwrite the last n_dup rows with copies of row 0 (inject redundancy)
        for i in range(n_dup):
            feats[19 - i] = base[0]
        vendis.append(vendi_score(normalized_gram(rff_features(feats, 1024, gamma=0.05, seed=0))))
    rho, _ = spearmanr(fractions, vendis)
    assert rho < -0.9, f"diversity should fall monotonically with redundancy, rho={rho}"


def test_greedy_dpp_picks_from_distinct_clusters():
    # 4 distinct clusters, 3 copies each -> budget 4 should pick one per cluster
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    feats = np.repeat(centers, 3, axis=0)  # 12 rows
    k = normalized_gram(rff_features(feats, 1024, gamma=0.05, seed=0))
    kept = greedy_dpp_map(k, 4)
    assert len(kept) == 4
    picked_centers = {tuple(feats[i]) for i in kept}
    assert len(picked_centers) == 4  # one from each cluster


def test_von_neumann_entropy_nonnegative_and_zero_for_rank1():
    feats = np.tile(np.array([1.0, 0.0]), (5, 1))
    k = normalized_gram(rff_features(feats, 256, gamma=0.5, seed=0))
    assert von_neumann_entropy(k) >= -1e-9
    assert von_neumann_entropy(k) < 1e-6  # rank-1 -> entropy ~ 0


def test_select_diverse_respects_budget_and_is_deterministic():
    rng = np.random.RandomState(2)
    feats = rng.randn(50, 12)
    kept1, info1 = select_diverse(feats, 10, n_rff=512, seed=7)
    kept2, info2 = select_diverse(feats, 10, n_rff=512, seed=7)
    assert kept1 == kept2  # deterministic
    assert len(kept1) == 10
    assert info1["vendi_full"] == info2["vendi_full"]
    assert info1["n_out"] == 10
