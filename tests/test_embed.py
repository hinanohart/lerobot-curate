import numpy as np

from lerobot_curate.diagnostics import doctor, doctor_text
from lerobot_curate.embed import Embedder, StubEmbedder, resolve_embedder
from lerobot_curate.ir import EmbeddingSource


def test_stub_deterministic_and_content_sensitive():
    emb = StubEmbedder(dim=32, seed=0)
    rng = np.random.RandomState(0)
    img_a = rng.rand(16, 16, 3)
    img_b = rng.rand(16, 16, 3)
    v1 = emb.embed_images([img_a, img_b])
    v2 = emb.embed_images([img_a, img_b])
    assert v1.shape == (2, 32)
    assert np.array_equal(v1, v2)  # deterministic
    # identical image -> identical embedding
    assert np.allclose(emb.embed_images([img_a])[0], v1[0])
    # different images -> different embedding
    assert not np.allclose(v1[0], v1[1])
    # L2 normalized
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0)


def test_stub_text_head():
    emb = StubEmbedder(dim=16)
    assert emb.has_text_head
    t = emb.embed_text(["pick cube", "open drawer", "pick cube"])
    assert t is not None and t.shape == (3, 16)
    assert np.allclose(t[0], t[2])  # same text -> same vector
    assert not np.allclose(t[0], t[1])
    assert emb.source is EmbeddingSource.STUB


def test_stub_empty_inputs():
    emb = StubEmbedder(dim=8)
    assert emb.embed_images([]).shape == (0, 8)
    assert emb.embed_text([]).shape == (0, 8)


def test_resolve_embedder_explicit_and_protocol():
    emb = resolve_embedder("stub")
    assert isinstance(emb, StubEmbedder)
    assert isinstance(emb, Embedder)  # runtime_checkable protocol


def test_resolve_embedder_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        resolve_embedder("does-not-exist")


def test_doctor_reports_stub_available():
    statuses = {s.name: s for s in doctor()}
    assert statuses["embedder:stub"].available is True
    txt = doctor_text().lower()
    assert "download on demand" in txt and "never bundled" in txt
