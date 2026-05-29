"""Diversity scoring and budget subset selection (pure numpy + scipy).

Pipeline (FAKTUAL-style, CPU reference implementation):

    signature features  -> random Fourier features (RBF approx)
                         -> normalized Gram matrix (unit diagonal)
                         -> von-Neumann entropy of K/n
                         -> Vendi score = exp(entropy)   (effective # distinct)
                         -> greedy DPP MAP budget subset (max log-det)

All deterministic given a seed. ``O(n^2)`` in the number of episodes, which is
why selection is performed **within a single dataset** (a few hundred to a few
thousand episodes); cross-dataset coverage is deferred to v0.1.1.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def median_heuristic_gamma(features: np.ndarray) -> float:
    """RBF bandwidth ``gamma`` from the median squared pairwise distance."""
    features = np.asarray(features, dtype=float)
    if features.shape[0] < 2:
        return 1.0
    d2 = pdist(features, metric="sqeuclidean")
    med = float(np.median(d2))
    if med <= 0.0:
        return 1.0
    return 1.0 / med


def rff_features(features: np.ndarray, n_features: int, gamma: float, seed: int) -> np.ndarray:
    """Random Fourier features approximating the RBF kernel ``exp(-gamma||x-y||^2)``."""
    features = np.asarray(features, dtype=float)
    d = int(features.shape[1])
    rng = np.random.RandomState(seed)
    w = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(d, n_features))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    proj = features @ w + b
    phi: np.ndarray = np.sqrt(2.0 / n_features) * np.cos(proj)
    return phi


def normalized_gram(phi: np.ndarray) -> np.ndarray:
    """Gram matrix ``phi phi^T`` rescaled to unit diagonal (a correlation matrix)."""
    k = phi @ phi.T
    diag = np.sqrt(np.clip(np.diag(k), 1e-12, None))
    k = k / np.outer(diag, diag)
    # numerical symmetry + clip into [-1, 1]
    k = 0.5 * (k + k.T)
    out: np.ndarray = np.clip(k, -1.0, 1.0)
    return out


def von_neumann_entropy(k: np.ndarray) -> float:
    """von-Neumann entropy of ``K / n`` (``K`` a unit-diagonal PSD kernel)."""
    n = k.shape[0]
    if n == 0:
        return 0.0
    w = np.linalg.eigvalsh(k / n)
    w = w[w > 1e-12]
    if w.size == 0:
        return 0.0
    return float(-(w * np.log(w)).sum())


def vendi_score(k: np.ndarray) -> float:
    """Vendi score = ``exp(von-Neumann entropy)``: effective number of distinct items."""
    if k.shape[0] == 0:
        return 0.0
    return float(np.exp(von_neumann_entropy(k)))


def greedy_dpp_map(k: np.ndarray, budget: int, eps: float = 1e-10) -> list[int]:
    """Fast greedy DPP MAP inference (Chen et al., 2018): maximize ``log det K_S``.

    Returns up to ``budget`` indices forming a high-diversity subset.
    """
    n = k.shape[0]
    budget = min(budget, n)
    if budget <= 0:
        return []
    cis = np.zeros((budget, n))
    di2 = np.clip(np.diag(k).astype(float).copy(), 0.0, None)
    selected: list[int] = []
    j = int(np.argmax(di2))
    selected.append(j)
    for _ in range(1, budget):
        m = len(selected) - 1
        ci_opt = cis[:m, j]
        di_opt = np.sqrt(di2[j]) if di2[j] > 0 else eps
        elements = k[j, :]
        eis = (elements - cis[:m, :].T @ ci_opt) / di_opt
        cis[m, :] = eis
        di2 = di2 - eis**2
        di2[selected] = -np.inf
        j = int(np.argmax(di2))
        if di2[j] < eps:
            break
        selected.append(j)
    return sorted(selected)


def select_diverse(
    features: np.ndarray,
    budget: int,
    *,
    n_rff: int = 512,
    gamma: float | None = None,
    seed: int = 0,
) -> tuple[list[int], dict[str, float | int]]:
    """Select a diverse budget-``budget`` subset of episodes from their features.

    Returns ``(kept_indices, info)`` where ``info`` records Vendi before/after,
    the bandwidth and RFF dimension used.
    """
    features = np.asarray(features, dtype=float)
    n = features.shape[0]
    if gamma is None:
        gamma = median_heuristic_gamma(features)
    phi = rff_features(features, n_rff, gamma, seed)
    k = normalized_gram(phi)
    kept = greedy_dpp_map(k, budget)
    vendi_subset = vendi_score(k[np.ix_(kept, kept)]) if kept else 0.0
    info: dict[str, float | int] = {
        "vendi_full": vendi_score(k),
        "vendi_subset": vendi_subset,
        "gamma": float(gamma),
        "n_rff": int(n_rff),
        "n_in": int(n),
        "n_out": len(kept),
    }
    return kept, info


def diversity_of(
    features: np.ndarray, *, n_rff: int = 512, gamma: float | None = None, seed: int = 0
) -> float:
    """Vendi score of a feature set (effective number of distinct trajectories)."""
    features = np.asarray(features, dtype=float)
    if features.shape[0] == 0:
        return 0.0
    if gamma is None:
        gamma = median_heuristic_gamma(features)
    phi = rff_features(features, n_rff, gamma, seed)
    return vendi_score(normalized_gram(phi))
