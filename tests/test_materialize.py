from lerobot_curate.curate import CurateConfig, curate
from lerobot_curate.embed import StubEmbedder
from lerobot_curate.io import make_synthetic_v3, materialize, open_dataset


def test_materialize_derived_dataset(tmp_path):
    src = tmp_path / "src"
    make_synthetic_v3(src, n_episodes=10, n_frames=8, dup_groups=[[0, 1]], seed=0)
    ds = open_dataset(str(src))
    result, report = curate(ds, StubEmbedder(dim=32), CurateConfig(budget=4, seed=0))

    out = tmp_path / "derived"
    mat = materialize(ds, result, str(out), report)
    assert mat.n_episodes == result.n_kept
    assert (out / "meta" / "info.json").exists()
    assert (out / "meta" / "episodes.jsonl").exists()
    assert (out / "curation_report.json").exists()

    # reopen derived dataset: episodes re-indexed contiguously 0..n-1
    derived = open_dataset(str(out))
    refs = derived.episode_refs()
    assert len(refs) == result.n_kept
    assert [r.episode_index for r in refs] == list(range(result.n_kept))
    info = derived.info()
    assert info["derived_by"] == "lerobot-curate"
    assert info["total_episodes"] == result.n_kept

    # derived report carries the scope disclaimer
    import json

    rep = json.loads((out / "curation_report.json").read_text())
    assert any("does NOT reproduce" in d for d in rep["disclaimers"])


def test_materialize_frames_readable(tmp_path):
    src = tmp_path / "src"
    make_synthetic_v3(src, n_episodes=6, n_frames=6, seed=1)
    ds = open_dataset(str(src))
    result, report = curate(ds, StubEmbedder(dim=16), CurateConfig(budget=3, seed=0))
    out = tmp_path / "derived"
    materialize(ds, result, str(out), report)
    derived = open_dataset(str(out))
    frames = derived.episode_frames(0, max_frames=4)
    assert frames.ndim == 4
