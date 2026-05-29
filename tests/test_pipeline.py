import json

from lerobot_curate.curate import CurateConfig, curate
from lerobot_curate.embed import StubEmbedder
from lerobot_curate.io import make_synthetic_v3, open_dataset
from lerobot_curate.ir import CurationReport, DropReason, SelectionResult


def test_pipeline_drops_duplicates(tmp_path):
    make_synthetic_v3(tmp_path, n_episodes=8, n_frames=8, dup_groups=[[0, 1], [2, 3]], seed=0)
    ds = open_dataset(str(tmp_path))
    result, report = curate(ds, StubEmbedder(dim=32), CurateConfig(seed=0))
    assert isinstance(result, SelectionResult)
    assert isinstance(report, CurationReport)
    # at least the injected duplicates are detected and dropped
    near_dup_drops = [ep for ep, r in result.drop_reasons.items() if r is DropReason.NEAR_DUPLICATE]
    assert len(near_dup_drops) >= 2
    assert report.n_episodes_in == 8
    assert report.n_episodes_out == result.n_kept
    assert len(report.dedup_pairs) >= 2
    # kept and dropped are disjoint and cover everything
    assert set(result.kept_episode_ids).isdisjoint(result.drop_reasons)


def test_pipeline_budget_respected(tmp_path):
    make_synthetic_v3(tmp_path, n_episodes=12, n_frames=8, seed=1)
    ds = open_dataset(str(tmp_path))
    result, report = curate(ds, StubEmbedder(dim=32), CurateConfig(budget=4, seed=0))
    assert result.n_kept <= 4
    assert report.n_episodes_out <= 4
    assert "feature_mode" in report.diversity
    # synthetic episodes carry state+action -> signature features (not embedding fallback)
    assert report.provenance["feature_mode"] == "signature"
    assert report.diversity["feature_mode"] == "signature"


def test_pipeline_report_json_roundtrips(tmp_path):
    make_synthetic_v3(tmp_path, n_episodes=6, n_frames=6, seed=2)
    ds = open_dataset(str(tmp_path))
    _, report = curate(ds, StubEmbedder(dim=16), CurateConfig(budget=3, seed=0))
    s = report.to_json()
    back = CurationReport.from_dict(json.loads(s))
    assert back.n_episodes_in == 6
    assert back.provenance["bytes_downloaded"] == 0  # local -> nothing downloaded
    assert any("does NOT reproduce" in d for d in back.disclaimers)


def test_pipeline_deterministic(tmp_path):
    make_synthetic_v3(tmp_path, n_episodes=10, n_frames=8, seed=3)
    ds = open_dataset(str(tmp_path))
    r1, _ = curate(ds, StubEmbedder(dim=32), CurateConfig(budget=5, seed=0))
    r2, _ = curate(ds, StubEmbedder(dim=32), CurateConfig(budget=5, seed=0))
    assert r1.kept_episode_ids == r2.kept_episode_ids


def test_pipeline_no_budget_keeps_survivors(tmp_path):
    make_synthetic_v3(tmp_path, n_episodes=5, n_frames=6, seed=4)
    ds = open_dataset(str(tmp_path))
    result, report = curate(ds, StubEmbedder(dim=16), CurateConfig(budget=None, seed=0))
    # distinct episodes, no budget -> nothing dropped for diversity
    assert all(r is not DropReason.LOW_DIVERSITY for r in result.drop_reasons.values())
    assert "vendi_full" in report.diversity
