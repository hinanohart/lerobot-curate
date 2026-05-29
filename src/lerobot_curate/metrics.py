"""Algorithm-correctness metrics (anti-theater, metric-independent ground truth).

These measure whether the SELECTION ALGORITHMS behave correctly against an
injected, known answer:

* dedup recall/precision on injected exact-duplicate episodes,
* mislabel precision/recall on injected label swaps (with aligned vs orthogonal
  embeddings so there is a known answer),
* monotone decrease of Vendi diversity under injected redundancy,
* near-zero false-positive rate on clean data.

They explicitly do NOT measure downstream policy-training performance — that
requires GPU policy training and is out of scope (see README scope disclaimer).
All functions are deterministic given a seed.
"""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .core.diversity import normalized_gram, rff_features, vendi_score
from .core.mislabel import MislabelItem, detect_mislabels
from .curate import CurateConfig, curate
from .embed import StubEmbedder
from .io import make_synthetic_v3, open_dataset
from .ir import DropReason, MislabelStatus


def dedup_metrics(seed: int = 0) -> dict[str, float]:
    """Recall/precision of near-duplicate detection on injected exact duplicates."""
    with tempfile.TemporaryDirectory() as d:
        gt = make_synthetic_v3(
            d, n_episodes=14, n_frames=8, dup_groups=[[0, 1, 2], [6, 7]], seed=seed
        )
        ds = open_dataset(d)
        result, _ = curate(
            ds, StubEmbedder(dim=32), CurateConfig(budget=None, drop_suboptimal=False, seed=seed)
        )
        injected = {b for _, b in gt.dup_pairs}  # non-leader members must be dropped
        detected = {ep for ep, r in result.drop_reasons.items() if r is DropReason.NEAR_DUPLICATE}
        tp = len(injected & detected)
        recall = tp / len(injected) if injected else 1.0
        precision = tp / len(detected) if detected else 1.0
        return {"dedup_recall": recall, "dedup_precision": precision}


def mislabel_metrics(seed: int = 0, n: int = 20, n_mis: int = 4) -> dict[str, float]:
    """Precision/recall of mislabel detection with a known set of swapped labels.

    Correct episodes have text==image (cosine 1); mislabeled episodes use a text
    vector orthogonal to the image (cosine ~0), so the answer is known.
    """
    rng = np.random.RandomState(seed)
    dim = 16
    half = dim // 2
    items: list[MislabelItem] = []
    truth: set[int] = set()
    for ep in range(n):
        img = np.zeros(dim)
        img[:half] = rng.randn(half)
        if ep < n_mis:
            txt = np.zeros(dim)
            txt[half:] = rng.randn(half)  # orthogonal to img -> cosine 0
            truth.add(ep)
        else:
            txt = img.copy()  # aligned -> cosine 1
        items.append(MislabelItem(ep, img, txt, "perform the labeled task"))
    q = min(0.5, n_mis / n + 0.1)
    flags = detect_mislabels(items, q=q, abs_floor=0.10)
    suspected = {f.episode_index for f in flags if f.status is MislabelStatus.SUSPECTED}
    tp = len(suspected & truth)
    precision = tp / len(suspected) if suspected else 1.0
    recall = tp / len(truth) if truth else 1.0
    return {"mislabel_precision": precision, "mislabel_recall": recall}


def diversity_monotonicity(seed: int = 0) -> dict[str, float | bool]:
    """Vendi must fall monotonically as redundancy is injected (Spearman rho)."""
    rng = np.random.RandomState(seed)
    base = rng.randn(20, 8) * 5.0
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8]
    vendis = []
    for frac in fractions:
        feats = base.copy()
        n_dup = int(frac * 20)
        for i in range(n_dup):
            feats[19 - i] = base[0]
        vendis.append(
            vendi_score(normalized_gram(rff_features(feats, 1024, gamma=0.05, seed=seed)))
        )
    rho, _ = spearmanr(fractions, vendis)
    return {"diversity_spearman_rho": float(rho), "diversity_monotonic": bool(rho < -0.9)}


def negative_fpr(seed: int = 0) -> dict[str, float]:
    """False-positive rate on clean data (no dups, aligned labels)."""
    # dedup FPR: distinct episodes -> no near-duplicates
    with tempfile.TemporaryDirectory() as d:
        make_synthetic_v3(d, n_episodes=12, n_frames=8, seed=seed)  # all distinct
        ds = open_dataset(d)
        result, _ = curate(
            ds, StubEmbedder(dim=32), CurateConfig(budget=None, drop_suboptimal=False, seed=seed)
        )
        n = max(1, len(ds.episode_refs()))
        false_dups = sum(1 for r in result.drop_reasons.values() if r is DropReason.NEAR_DUPLICATE)
        dedup_fpr = false_dups / n

    # mislabel FPR: aligned (coherent) embeddings -> nothing flagged
    rng = np.random.RandomState(seed + 1)
    items = []
    for ep in range(15):
        v = rng.randn(12)
        items.append(MislabelItem(ep, v, v.copy(), "a perfectly valid task description"))
    flags = detect_mislabels(items, q=0.05, abs_floor=0.10)
    mis_fpr = sum(1 for f in flags if f.status is MislabelStatus.SUSPECTED) / len(items)

    return {
        "negative_fpr_dedup": dedup_fpr,
        "negative_fpr_mislabel": mis_fpr,
        "negative_fpr": max(dedup_fpr, mis_fpr),
    }


def compute_metrics(seed: int = 0) -> dict[str, Any]:
    """Run every algorithm-correctness metric and return a flat dict."""
    out: dict[str, Any] = {}
    out.update(dedup_metrics(seed))
    out.update(mislabel_metrics(seed))
    out.update(diversity_monotonicity(seed))
    out.update(negative_fpr(seed))
    out["dataset"] = {
        "id": "synthetic-injected",
        "n": 14,
        "mode": "synthetic",
        "source": "make_synthetic_v3",
    }
    out["disclaimer_required"] = True  # synthetic-only validation
    return out
