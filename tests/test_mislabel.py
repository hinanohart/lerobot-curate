import numpy as np

from lerobot_curate.core.mislabel import MislabelItem, detect_mislabels, usable_task
from lerobot_curate.ir import MislabelStatus


def test_usable_task():
    assert usable_task("pick up the cube")
    assert not usable_task(None)
    assert not usable_task("")
    assert not usable_task("  ")
    assert not usable_task("n/a")
    assert not usable_task("TODO")
    assert not usable_task("task")
    assert not usable_task("..")


def test_detect_mislabels_flags_swapped_label():
    # 9 aligned episodes (image==text) + 1 swapped (orthogonal) -> swapped suspected
    dim = 8
    aligned_vec = np.zeros(dim)
    aligned_vec[0] = 1.0
    items = []
    for ep in range(9):
        items.append(MislabelItem(ep, aligned_vec.copy(), aligned_vec.copy(), "pick the cube"))
    wrong = np.zeros(dim)
    wrong[1] = 1.0  # orthogonal to aligned image
    items.append(MislabelItem(9, aligned_vec.copy(), wrong, "open the drawer"))

    flags = {f.episode_index: f for f in detect_mislabels(items, q=0.05, abs_floor=0.10)}
    assert flags[9].status is MislabelStatus.SUSPECTED
    assert all(flags[e].status is MislabelStatus.OK for e in range(9))


def test_detect_mislabels_degrade_first_class():
    dim = 4
    v = np.ones(dim)
    items = [
        MislabelItem(0, v.copy(), v.copy(), "valid task here"),
        MislabelItem(1, v.copy(), None, "valid task but no text head"),
        MislabelItem(2, v.copy(), v.copy(), ""),  # empty task
    ]
    flags = {f.episode_index: f for f in detect_mislabels(items)}
    assert flags[1].status is MislabelStatus.NOT_EVALUATED
    assert flags[2].status is MislabelStatus.NOT_EVALUATED
    # the only scored one cannot be suspected against itself alone unless below floor
    assert flags[0].status in (MislabelStatus.OK, MislabelStatus.SUSPECTED)


def test_detect_mislabels_clean_dataset_no_false_positive():
    # uniformly high-cosine dataset: the AND-with-floor rule prevents flagging
    dim = 6
    rng = np.random.RandomState(0)
    items = []
    for ep in range(12):
        x = rng.randn(dim)
        items.append(
            MislabelItem(ep, x, x.copy(), "do the thing properly")
        )  # identical -> cosine 1
    flags = detect_mislabels(items, q=0.05, abs_floor=0.10)
    assert all(f.status is MislabelStatus.OK for f in flags)
