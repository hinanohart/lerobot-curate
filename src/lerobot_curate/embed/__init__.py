"""Embedding backends.

Default resolution is honest: the real CPU ONNX embedder is preferred, but if
its optional dependency (onnxruntime) or weights are unavailable we fall back to
the deterministic stub and say so via ``doctor`` rather than failing silently.
"""

from __future__ import annotations

import os

from .base import Embedder
from .stub import StubEmbedder

__all__ = ["Embedder", "StubEmbedder", "resolve_embedder"]


def resolve_embedder(name: str | None = None, **kwargs: object) -> Embedder:
    """Return an embedder by name.

    ``name`` may be ``"stub"``, ``"local-onnx"``, ``"hf-api"`` or ``None``.
    When ``None`` the ``LEROBOT_CURATE_EMBEDDER`` env var is consulted, then we
    try ``local-onnx`` and fall back to ``stub`` if onnxruntime is missing.
    """
    name = name or os.environ.get("LEROBOT_CURATE_EMBEDDER")
    if name in (None, "auto"):
        try:
            import onnxruntime  # noqa: F401

            name = "local-onnx"
        except ImportError:
            name = "stub"

    if name == "stub":
        return StubEmbedder(**kwargs)  # type: ignore[arg-type]
    if name == "local-onnx":
        from .local_onnx import LocalOnnxEmbedder

        return LocalOnnxEmbedder(**kwargs)  # type: ignore[arg-type]
    if name == "hf-api":
        from .hf_api import HfApiEmbedder

        return HfApiEmbedder(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown embedder: {name!r}")
