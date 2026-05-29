from lerobot_curate.metrics import (
    compute_metrics,
    dedup_metrics,
    diversity_monotonicity,
    mislabel_metrics,
    negative_fpr,
)


def test_dedup_metrics_perfect_on_injected_dups():
    m = dedup_metrics(0)
    assert m["dedup_recall"] == 1.0
    assert m["dedup_precision"] == 1.0


def test_mislabel_metrics_perfect_on_known_swaps():
    m = mislabel_metrics(0)
    assert m["mislabel_precision"] == 1.0
    assert m["mislabel_recall"] == 1.0


def test_diversity_monotonic_under_redundancy():
    m = diversity_monotonicity(0)
    assert m["diversity_monotonic"] is True
    assert m["diversity_spearman_rho"] < -0.9


def test_negative_fpr_near_zero_on_clean_data():
    m = negative_fpr(0)
    assert m["negative_fpr_dedup"] == 0.0
    assert m["negative_fpr_mislabel"] == 0.0
    assert m["negative_fpr"] <= 0.05


def test_compute_metrics_full_dict():
    m = compute_metrics(0)
    for k in ("dedup_recall", "mislabel_precision", "negative_fpr", "diversity_monotonic"):
        assert m[k] is not None
    assert m["dataset"]["mode"] == "synthetic"
    assert m["disclaimer_required"] is True
