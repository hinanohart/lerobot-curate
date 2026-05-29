"""Opt-in Hugging Face Inference API embedder (OFF by default).

The free Inference tier cannot embed the thousands of frames a real curation run
needs, so this backend is opt-in only and is flagged ``degraded`` in reports. It
is exercised only under the ``hf_api`` test marker (requires credentials).
"""

from __future__ import annotations

import numpy as np

from ..ir import EmbeddingSource


class HfApiEmbedder:
    source: EmbeddingSource = EmbeddingSource.HF_API

    def __init__(
        self, model_id: str = "google/siglip-base-patch16-224", token: str | None = None
    ) -> None:
        self.model_id = model_id
        self._token = token
        self._client: object | None = None
        self._dim: int | None = None
        self.degraded = True  # always: free tier is not sized for full datasets

    def _ensure_client(self) -> object:
        if self._client is None:
            from huggingface_hub import InferenceClient

            # token=None lets huggingface_hub use the ambient login; we never read
            # or print the token value (R11).
            self._client = InferenceClient(model=self.model_id, token=self._token)
        return self._client

    @property
    def has_text_head(self) -> bool:
        return True

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("dim unknown until first embed call")
        return self._dim

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        out: np.ndarray = np.asarray(v / np.clip(n, 1e-12, None))
        return out

    def embed_text(self, texts: list[str]) -> np.ndarray | None:
        client = self._ensure_client()
        vecs = [np.asarray(client.feature_extraction(t), dtype=np.float32).ravel() for t in texts]  # type: ignore[attr-defined]
        out = self._l2(np.stack(vecs)) if vecs else np.zeros((0, 0))
        if out.size:
            self._dim = int(out.shape[1])
        return out

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError(
            "Image embedding over the HF Inference API is not enabled in a1; "
            "use the local-onnx backend for frames."
        )
