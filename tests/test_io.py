import numpy as np

from lerobot_curate.io import make_synthetic_v3, open_dataset


def test_synthetic_roundtrip(tmp_path):
    gt = make_synthetic_v3(
        tmp_path,
        n_episodes=8,
        n_frames=10,
        frame_hw=(16, 16),
        dup_groups=[[0, 1]],
        mislabel_eps=[2],
        empty_task_eps=[3],
        seed=0,
    )
    assert gt.dup_pairs == [(0, 1)]
    assert gt.mislabel_eps == [2]
    assert gt.empty_task_eps == [3]

    ds = open_dataset(str(tmp_path))
    assert ds.is_local()
    refs = ds.episode_refs()
    assert len(refs) == 8
    assert all(r.num_frames == 10 for r in refs)
    assert refs[0].fps == 10

    # frames shape and dtype
    f0 = ds.episode_frames(0, max_frames=4)
    assert f0.ndim == 4 and f0.shape[1:] == (16, 16, 3)
    assert f0.shape[0] == 4

    # duplicate episodes have identical frames
    f1 = ds.episode_frames(1, max_frames=4)
    assert np.allclose(f0, f1)

    # signature path = state(4) + action(2)
    p0 = ds.episode_path(0)
    assert p0.shape == (10, 6)

    # task assignment: empty for ep3, mislabel differs from base for ep2
    assert ds.episode_task(3) == ""
    assert ds.episode_task(2) != "stack the blocks"  # base would be index 2


def test_synthetic_distinct_episodes_differ(tmp_path):
    make_synthetic_v3(tmp_path, n_episodes=4, n_frames=8, seed=1)
    ds = open_dataset(str(tmp_path))
    a = ds.episode_frames(0, max_frames=8)
    b = ds.episode_frames(2, max_frames=8)
    assert not np.allclose(a, b)


def test_info_features(tmp_path):
    make_synthetic_v3(tmp_path, n_episodes=2, n_frames=5, state_dim=4, action_dim=2)
    ds = open_dataset(str(tmp_path))
    info = ds.info()
    assert info["codebase_version"] == "v3.0"
    assert info["features"]["observation.state"]["shape"] == [4]
