"""Embedder protocol shared by all backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..ir import EmbeddingSource


@runtime_checkable
class Embedder(Protocol):
    """A CPU embedder mapping frames (and optionally task text) to vectors.

    ``embed_images`` returns an array of shape ``(n, dim)``. ``embed_text``
    returns ``(n, dim)`` when the backend has a text head, else ``None`` — callers
    must treat ``None`` as "cross-modal mislabel cannot be evaluated" rather than
    assuming alignment.
    """

    source: EmbeddingSource
    model_id: str

    @property
    def dim(self) -> int: ...

    @property
    def has_text_head(self) -> bool: ...

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray: ...

    def embed_text(self, texts: list[str]) -> np.ndarray | None: ...
