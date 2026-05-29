import json

import numpy as np
import pytest

from lerobot_curate.ir import (
    SCOPE_DISCLAIMER,
    SIGNATURE_SCOPE,
    CurationReport,
    DropReason,
    EmbeddingSource,
    EpisodeRef,
    EpisodeSignature,
    FrameEmbedding,
    MislabelFlag,
    MislabelStatus,
    SelectionResult,
)


def test_episode_ref_validates():
    ref = EpisodeRef(repo_id="x/y", episode_index=3, num_frames=120, fps=30.0, task="pick cube")
    assert ref.episode_index == 3
    with pytest.raises(ValueError):
        EpisodeRef(repo_id="x/y", episode_index=-1, num_frames=10)
    with pytest.raises(ValueError):
        EpisodeRef(repo_id="x/y", episode_index=0, num_frames=-5)


def test_frame_embedding_shapes_and_props():
    v = np.random.RandomState(0).randn(5, 16).astype(np.float32)
    fe = FrameEmbedding(episode_index=0, vectors=v, source=EmbeddingSource.STUB, model_id="stub")
    assert fe.dim == 16
    assert fe.mean_vector.shape == (16,)
    with pytest.raises(ValueError):
        FrameEmbedding(episode_index=0, vectors=v[0], source=EmbeddingSource.STUB, model_id="s")
    # text head dim must match
    with pytest.raises(ValueError):
        FrameEmbedding(
            episode_index=0,
            vectors=v,
            source=EmbeddingSource.STUB,
            model_id="s",
            text_vector=np.zeros(8, dtype=np.float32),
        )
    fe2 = FrameEmbedding(
        episode_index=1,
        vectors=v,
        source=EmbeddingSource.STUB,
        model_id="s",
        text_vector=np.zeros(16, dtype=np.float32),
    )
    assert fe2.text_vector is not None


def test_signature_scope_invariant():
    sig = EpisodeSignature(episode_index=0, coeffs=np.zeros(10), depth=3, rff_dim=512)
    assert sig.scope == SIGNATURE_SCOPE
    with pytest.raises(ValueError):
        EpisodeSignature(episode_index=0, coeffs=np.zeros(10), depth=3, rff_dim=512, scope="cross")
    with pytest.raises(ValueError):
        EpisodeSignature(episode_index=0, coeffs=np.zeros((2, 2)), depth=3, rff_dim=512)
    with pytest.raises(ValueError):
        EpisodeSignature(episode_index=0, coeffs=np.zeros(10), depth=0, rff_dim=512)


def test_selection_result_no_overlap():
    sr = SelectionResult(
        kept_episode_ids=[0, 2],
        drop_reasons={1: DropReason.NEAR_DUPLICATE, 3: DropReason.LOW_DIVERSITY},
    )
    assert sr.n_kept == 2
    assert sr.n_dropped == 2
    with pytest.raises(ValueError):
        SelectionResult(kept_episode_ids=[0, 1], drop_reasons={1: DropReason.SUBOPTIMAL})


def test_mislabel_flag_degrade_first_class():
    flag = MislabelFlag(episode_index=4, status=MislabelStatus.NOT_EVALUATED, reason="empty task")
    d = flag.to_dict()
    assert d["status"] == "not-evaluated"
    assert d["ep"] == 4


def test_curation_report_roundtrip_json():
    rep = CurationReport(
        source_repo="lerobot/pusht",
        tool_version="0.1.0a1",
        source_revision="abc123",
        embedder={"id": "stub", "path": "stub"},
        n_episodes_in=10,
        n_episodes_out=4,
        kept_episode_ids=[0, 1, 2, 3],
        drop_reasons={4: "near-duplicate", 5: "low-diversity"},
        dedup_pairs=[(0, 4)],
        diversity={"vendi": 3.14},
        mislabel_flags=[MislabelFlag(6, MislabelStatus.SUSPECTED, 0.05, "low cosine")],
        provenance={"bytes_downloaded": 0, "streamed": True, "seed": 0},
    )
    s = rep.to_json()
    back = CurationReport.from_dict(json.loads(s))
    assert back.source_repo == "lerobot/pusht"
    assert back.n_episodes_out == 4
    assert back.drop_reasons[4] == "near-duplicate"
    assert back.dedup_pairs == [(0, 4)]
    assert back.mislabel_flags[0].status is MislabelStatus.SUSPECTED
    # default disclaimer always present
    assert SCOPE_DISCLAIMER in back.disclaimers
    # JSON keys are strings
    assert json.loads(s)["drop_reasons"]["4"] == "near-duplicate"


def test_disclaimer_contains_required_phrase():
    assert "does NOT reproduce their reported policy-performance gains" in SCOPE_DISCLAIMER
