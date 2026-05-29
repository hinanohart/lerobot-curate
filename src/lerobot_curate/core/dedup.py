"""Embedding dedup and model-free suboptimal-episode proxies.

Dedup follows the SemDeDup recipe: cluster episode-level embeddings with k-means,
then within each cluster drop episodes whose cosine similarity to an already-kept
representative exceeds a threshold. This is a CPU reference implementation; it
does not reproduce any policy-training numbers.

The suboptimal proxies (embedding-jump p95, action reversal ratio, action
quiescence) are heuristics for "this episode looks erratic or idle". They are
proxies only — not validated against downstream task outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    out: np.ndarray = np.asarray(x / np.clip(n, 1e-12, None))
    return out


def find_near_duplicates(
    embeddings: np.ndarray,
    cosine_threshold: float = 0.99,
    n_clusters: int | None = None,
    seed: int = 0,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return ``(episodes_to_drop, dup_pairs)``.

    ``dup_pairs`` are ``(kept_representative, dropped)`` index pairs.
    """
    n = int(embeddings.shape[0])
    if n <= 1:
        return [], []
    x = _normalize(np.asarray(embeddings, dtype=float))
    k = n_clusters if n_clusters is not None else max(1, int(round(np.sqrt(n))))
    k = min(k, n)
    if k >= 2:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x)
    else:
        labels = np.zeros(n, dtype=int)

    drop: list[int] = []
    pairs: list[tuple[int, int]] = []
    for c in np.unique(labels):
        members = [int(i) for i in np.where(labels == c)[0]]
        reps: list[int] = []
        for i in members:
            dup_of: int | None = None
            for r in reps:
                if float(x[i] @ x[r]) >= cosine_threshold:
                    dup_of = r
                    break
            if dup_of is None:
                reps.append(i)
            else:
                drop.append(i)
                pairs.append((dup_of, i))
    return sorted(drop), pairs


@dataclass
class SuboptimalScores:
    jump_p95: float
    reversal_ratio: float
    quiescent_frac: float


@dataclass
class SuboptimalThresholds:
    jump_p95: float = float("inf")
    reversal_ratio: float = float("inf")
    quiescent_frac: float = 0.9


def suboptimal_scores(
    frame_vectors: np.ndarray,
    actions: np.ndarray | None = None,
    quiescent_eps: float = 1e-3,
) -> SuboptimalScores:
    """Model-free proxies for one episode.

    ``frame_vectors`` is ``(k, dim)`` representative-frame embeddings; ``actions``
    is ``(T, action_dim)`` if available.
    """
    fv = np.asarray(frame_vectors, dtype=float)
    if fv.shape[0] > 1:
        jumps = np.linalg.norm(np.diff(fv, axis=0), axis=1)
        jp95 = float(np.percentile(jumps, 95))
    else:
        jp95 = 0.0

    reversal = 0.0
    quiescent = 0.0
    if actions is not None:
        a = np.asarray(actions, dtype=float)
        if a.ndim == 2 and a.shape[0] > 1:
            signs = np.sign(a)
            sign_changes = int(np.sum(np.diff(signs, axis=0) != 0))
            denom = max(1, (a.shape[0] - 1) * a.shape[1])
            reversal = float(sign_changes / denom)
            quiescent = float(np.mean(np.linalg.norm(a, axis=1) < quiescent_eps))
    return SuboptimalScores(jump_p95=jp95, reversal_ratio=reversal, quiescent_frac=quiescent)


def is_suboptimal(scores: SuboptimalScores, thr: SuboptimalThresholds) -> bool:
    return (
        scores.jump_p95 > thr.jump_p95
        or scores.reversal_ratio > thr.reversal_ratio
        or scores.quiescent_frac > thr.quiescent_frac
    )
