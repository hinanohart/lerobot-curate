"""Truncated path-signature features in pure numpy (torch-free).

The path signature is the central object of rough-path theory: for a path
``X: [0, T] -> R^d`` it is the sequence of iterated integrals. For a
piecewise-linear path (the discrete case here) the signature is computed exactly
by Chen's identity: the signature of a concatenation of segments is the tensor
product (in the truncated tensor algebra) of the per-segment signatures, and a
single linear segment with increment ``Δ`` has signature ``exp(Δ)`` =
``(1, Δ, Δ⊗Δ/2!, Δ⊗Δ⊗Δ/3!, ...)``.

We represent a depth-``N`` signature as a list ``sig`` of length ``N+1`` where
``sig[k]`` is an array of shape ``(d,) * k`` (``sig[0]`` is the scalar ``1``).

Two standard augmentations are provided:

* **time augmentation** prepends a monotone time coordinate, making the signature
  sensitive to the speed/monotonicity of the path;
* **lead-lag** doubles the dimension and exposes the quadratic variation (the
  level-2 area term), which is what makes the signature kernel discriminative for
  noisy/oscillatory trajectories.

All functions are deterministic and depend only on numpy.
"""

from __future__ import annotations

import numpy as np


def segment_signature(delta: np.ndarray, depth: int) -> list[np.ndarray]:
    """Signature of a single linear segment with increment ``delta``.

    ``sig[k] = delta^{⊗k} / k!``.
    """
    delta = np.asarray(delta, dtype=float)
    sig: list[np.ndarray] = [np.array(1.0)]
    term: np.ndarray = np.array(1.0)
    for k in range(1, depth + 1):
        term = np.multiply.outer(term, delta) / k
        sig.append(term)
    return sig


def chen_product(
    a: list[np.ndarray], b: list[np.ndarray], depth: int, dim: int
) -> list[np.ndarray]:
    """Tensor-algebra (Chen) product of two truncated signatures, to ``depth``.

    ``c[k] = sum_{i=0}^{k} a[i] ⊗ b[k-i]``.
    """
    c: list[np.ndarray] = []
    for k in range(depth + 1):
        acc = np.zeros((dim,) * k)
        for i in range(k + 1):
            acc = acc + np.multiply.outer(a[i], b[k - i])
        c.append(acc)
    return c


def path_signature(points: np.ndarray, depth: int) -> list[np.ndarray]:
    """Exact truncated signature of the piecewise-linear path through ``points``.

    ``points`` has shape ``(L+1, d)``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"points must be 2D (L+1, d), got shape {points.shape}")
    dim = int(points.shape[1])
    deltas = np.diff(points, axis=0)
    if deltas.shape[0] == 0:
        return [np.array(1.0)] + [np.zeros((dim,) * k) for k in range(1, depth + 1)]
    sig = segment_signature(deltas[0], depth)
    for i in range(1, deltas.shape[0]):
        sig = chen_product(sig, segment_signature(deltas[i], depth), depth, dim)
    return sig


def time_augment(points: np.ndarray) -> np.ndarray:
    """Prepend a monotone time coordinate in ``[0, 1]`` to each point."""
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    t = np.linspace(0.0, 1.0, n).reshape(n, 1)
    return np.concatenate([t, points], axis=1)


def lead_lag_transform(points: np.ndarray) -> np.ndarray:
    """Lead-lag embedding: ``(L+1, d) -> (2L+1, 2d)``.

    ``Z_{2i} = (x_i, x_i)``, ``Z_{2i+1} = (x_{i+1}, x_i)``, ``Z_{2L} = (x_L, x_L)``.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    d = int(points.shape[1])
    if n == 0:
        return np.zeros((0, 2 * d))
    out = np.zeros((2 * (n - 1) + 1, 2 * d))
    for i in range(n - 1):
        out[2 * i, :d] = points[i]
        out[2 * i, d:] = points[i]
        out[2 * i + 1, :d] = points[i + 1]
        out[2 * i + 1, d:] = points[i]
    out[-1, :d] = points[-1]
    out[-1, d:] = points[-1]
    return out


def signature_features(
    points: np.ndarray,
    depth: int = 3,
    *,
    use_lead_lag: bool = True,
    use_time_aug: bool = True,
) -> np.ndarray:
    """Flattened signature feature vector (levels ``1..depth``, level 0 dropped).

    Applies lead-lag then time augmentation (deterministic order) before
    computing the signature, when enabled.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"points must be 2D (L+1, d), got shape {p.shape}")
    if p.shape[0] < 1:
        raise ValueError("need at least one point")
    if use_lead_lag:
        p = lead_lag_transform(p)
    if use_time_aug:
        p = time_augment(p)
    sig = path_signature(p, depth)
    return np.concatenate([np.asarray(sig[k]).ravel() for k in range(1, depth + 1)])


def signed_area(points: np.ndarray) -> float:
    """Signed area enclosed by a 2D path = ``0.5 * (S2[0,1] - S2[1,0])``."""
    points = np.asarray(points, dtype=float)
    if points.shape[1] != 2:
        raise ValueError("signed_area requires a 2D path")
    sig = path_signature(points, 2)
    s2 = sig[2]
    return 0.5 * float(s2[0, 1] - s2[1, 0])
