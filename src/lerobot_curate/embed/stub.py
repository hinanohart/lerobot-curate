"""Deterministic stub embedder (default in CI; no weights, no network).

Maps each image to a fixed-length descriptor (resampled pixel sequence) then a
fixed random projection, L2-normalized. Identical images produce identical
vectors and similar images produce similar vectors, so dedup/diversity behave
sensibly on synthetic data. Text is embedded by a content hash. This backend is
for plumbing and offline tests; it is NOT a substitute for a real vision encoder.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..ir import EmbeddingSource


class StubEmbedder:
    source: EmbeddingSource = EmbeddingSource.STUB
    model_id: str = "stub-deterministic-v1"

    def __init__(self, dim: int = 32, descriptor_len: int = 256, seed: int = 0) -> None:
        self._dim = int(dim)
        self._desc_len = int(descriptor_len)
        rng = np.random.RandomState(seed)
        self._proj = rng.randn(self._desc_len, self._dim)
        self._text_proj = rng.randn(64, self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def has_text_head(self) -> bool:
        return True

    def _descriptor(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float).ravel()
        if a.size == 0:
            a = np.zeros(1)
        idx = np.linspace(0, a.size - 1, self._desc_len)
        desc: np.ndarray = np.interp(idx, np.arange(a.size), a)
        return desc

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        out: np.ndarray = np.asarray(v / np.clip(n, 1e-12, None))
        return out

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim))
        desc = np.stack([self._descriptor(im) for im in images])
        return self._l2(desc @ self._proj)

    def embed_text(self, texts: list[str]) -> np.ndarray | None:
        if not texts:
            return np.zeros((0, self._dim))
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            h = hashlib.sha256((t or "").encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "little")
            out[i] = np.random.RandomState(seed).randn(64)
        return self._l2(out @ self._text_proj)
