import numpy as np

from lerobot_curate.core.dedup import (
    SuboptimalThresholds,
    find_near_duplicates,
    is_suboptimal,
    suboptimal_scores,
)


def test_find_near_duplicates_detects_exact_copy():
    rng = np.random.RandomState(0)
    base = rng.randn(5, 16)
    emb = np.vstack([base, base[0]])  # row 5 duplicates row 0
    drop, pairs = find_near_duplicates(emb, cosine_threshold=0.99, seed=0)
    assert 5 in drop
    assert any(b == 5 for _, b in pairs)
    # the kept representative of the dup is row 0
    assert any(a == 0 and b == 5 for a, b in pairs)


def test_find_near_duplicates_none_for_distinct():
    emb = np.eye(6) * 10.0  # mutually orthogonal -> no near-duplicates
    drop, pairs = find_near_duplicates(emb, cosine_threshold=0.99, seed=0)
    assert drop == []
    assert pairs == []


def test_find_near_duplicates_trivial_sizes():
    assert find_near_duplicates(np.zeros((0, 4))) == ([], [])
    assert find_near_duplicates(np.ones((1, 4))) == ([], [])


def test_suboptimal_quiescent_actions():
    fv = np.random.RandomState(1).randn(10, 8)
    actions = np.zeros((10, 2))  # no movement -> fully quiescent
    sc = suboptimal_scores(fv, actions)
    assert sc.quiescent_frac == 1.0
    assert is_suboptimal(sc, SuboptimalThresholds(quiescent_frac=0.9))


def test_suboptimal_active_episode_not_flagged_by_default():
    rng = np.random.RandomState(2)
    fv = np.cumsum(rng.randn(10, 8) * 0.1, axis=0)
    actions = rng.randn(10, 2)  # active
    sc = suboptimal_scores(fv, actions)
    assert sc.quiescent_frac < 0.5
    # default thresholds are conservative (inf jumps/reversals) -> not flagged
    assert not is_suboptimal(sc, SuboptimalThresholds())


def test_suboptimal_no_actions():
    fv = np.random.RandomState(3).randn(6, 8)
    sc = suboptimal_scores(fv, None)
    assert sc.reversal_ratio == 0.0
    assert sc.quiescent_frac == 0.0
    assert sc.jump_p95 >= 0.0
