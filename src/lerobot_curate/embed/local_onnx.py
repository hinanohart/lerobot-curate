"""Local CPU ONNX embedder: Xenova SigLIP-base int8 (vision + text head).

Weights are downloaded on demand via ``huggingface_hub`` to the local HF cache
and are NEVER bundled with this package nor committed to git. If onnxruntime is
not installed this module cannot be imported; callers should fall back to the
stub (``resolve_embedder`` does this automatically).

This backend talks to the network and to real weights, so it is exercised only
under the ``live`` test marker, never in CI. ``intra_op_num_threads=1`` is set
for run-to-run determinism on a fixed machine.
"""

from __future__ import annotations

import numpy as np

from ..ir import EmbeddingSource

_DEFAULT_REPO = "Xenova/siglip-base-patch16-224"
_VISION_FILE = "onnx/vision_model_quantized.onnx"
_TEXT_FILE = "onnx/text_model_quantized.onnx"
_IMG_SIZE = 224


class LocalOnnxEmbedder:
    source: EmbeddingSource = EmbeddingSource.LOCAL_ONNX

    def __init__(
        self,
        repo_id: str = _DEFAULT_REPO,
        vision_file: str = _VISION_FILE,
        text_file: str = _TEXT_FILE,
        revision: str | None = None,
    ) -> None:
        self.model_id = repo_id
        self._repo_id = repo_id
        self._vision_file = vision_file
        self._text_file = text_file
        self._revision = revision
        self._vsess: object | None = None
        self._tsess: object | None = None
        self._tokenizer: object | None = None
        self._dim: int | None = None
        self._text_ok: bool | None = None

    # -- lazy session construction ------------------------------------------
    def _ort_session(self, local_path: str) -> object:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        return ort.InferenceSession(local_path, sess_options=so, providers=["CPUExecutionProvider"])

    def _ensure_vision(self) -> None:
        if self._vsess is not None:
            return
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self._repo_id, self._vision_file, revision=self._revision)
        self._vsess = self._ort_session(path)

    def _ensure_text(self) -> bool:
        if self._text_ok is not None:
            return self._text_ok
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            tok_path = hf_hub_download(self._repo_id, "tokenizer.json", revision=self._revision)
            self._tokenizer = Tokenizer.from_file(tok_path)
            tpath = hf_hub_download(self._repo_id, self._text_file, revision=self._revision)
            self._tsess = self._ort_session(tpath)
            self._text_ok = True
        except Exception:  # noqa: BLE001 - any failure means no usable text head
            self._text_ok = False
        return self._text_ok

    @property
    def has_text_head(self) -> bool:
        return self._ensure_text()

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed_images([np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)])
        assert self._dim is not None
        return self._dim

    # -- preprocessing -------------------------------------------------------
    @staticmethod
    def _preprocess(images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        batch = np.zeros((len(images), 3, _IMG_SIZE, _IMG_SIZE), dtype=np.float32)
        for i, im in enumerate(images):
            arr = np.asarray(im)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            pil = (
                Image.fromarray(arr.astype(np.uint8)).convert("RGB").resize((_IMG_SIZE, _IMG_SIZE))
            )
            x = np.asarray(pil, dtype=np.float32) / 255.0
            x = (x - 0.5) / 0.5  # SigLIP normalization
            batch[i] = x.transpose(2, 0, 1)
        return batch

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        out: np.ndarray = np.asarray(v / np.clip(n, 1e-12, None))
        return out

    def _run(self, sess: object, feeds: dict[str, np.ndarray]) -> np.ndarray:
        out = sess.run(None, feeds)  # type: ignore[attr-defined]
        arr = np.asarray(out[0], dtype=np.float32)
        if arr.ndim == 3:  # last_hidden_state -> mean pool over tokens
            arr = arr.mean(axis=1)
        res: np.ndarray = np.asarray(arr, dtype=np.float32)
        return res

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim or 0))
        self._ensure_vision()
        assert self._vsess is not None
        feeds = {self._vsess.get_inputs()[0].name: self._preprocess(images)}  # type: ignore[attr-defined]
        emb = self._l2(self._run(self._vsess, feeds))
        self._dim = int(emb.shape[1])
        return emb

    def embed_text(self, texts: list[str]) -> np.ndarray | None:
        if not self._ensure_text():
            return None
        if not texts:
            return np.zeros((0, self._dim or 0))
        assert self._tokenizer is not None and self._tsess is not None
        encs = [self._tokenizer.encode(t or "") for t in texts]  # type: ignore[attr-defined]
        maxlen = max(len(e.ids) for e in encs)
        ids = np.zeros((len(texts), maxlen), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
        feeds = {self._tsess.get_inputs()[0].name: ids}  # type: ignore[attr-defined]
        return self._l2(self._run(self._tsess, feeds))
