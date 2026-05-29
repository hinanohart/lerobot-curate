"""Build a small synthetic LeRobot-v3-style dataset on local disk.

This is the offline ground-truth source for tests, the S6 correctness metrics,
and the end-to-end materialize smoke. The layout mirrors LeRobot v3 conventions
closely enough for our reader:

    root/meta/info.json          codebase_version, fps, feature shapes
    root/meta/episodes.jsonl     one row per episode: episode_index, length, tasks
    root/data/episode_NNNNNN.parquet   columns: observation.state, action, frame, timestamp

Ground-truth knobs let tests inject exact-duplicate episodes and mislabeled task
strings so that recall/precision/FPR can be measured against a known answer
(critic patch P5: metric-independent ground truth, no self-scoring).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v3.0"


@dataclass
class SyntheticGroundTruth:
    n_episodes: int
    frame_hw: tuple[int, int]
    dup_pairs: list[tuple[int, int]] = field(default_factory=list)
    mislabel_eps: list[int] = field(default_factory=list)
    empty_task_eps: list[int] = field(default_factory=list)
    tasks: dict[int, str] = field(default_factory=dict)


def _episode_frames(rng: np.random.RandomState, n_frames: int, hw: tuple[int, int]) -> np.ndarray:
    """A smooth synthetic episode: a moving Gaussian blob plus low noise."""
    h, w = hw
    frames = np.zeros((n_frames, h, w, 3), dtype=np.float32)
    cy, cx = rng.uniform(2, h - 2), rng.uniform(2, w - 2)
    dy, dx = rng.uniform(-1, 1), rng.uniform(-1, 1)
    color = rng.uniform(0.4, 1.0, size=3)
    ys, xs = np.mgrid[0:h, 0:w]
    for t in range(n_frames):
        y = cy + dy * t
        x = cx + dx * t
        blob = np.exp(-((ys - y) ** 2 + (xs - x) ** 2) / 4.0)
        frames[t] = (blob[..., None] * color)[None] + rng.normal(0, 0.01, (h, w, 3))
    return np.clip(frames, 0.0, 1.0)


def make_synthetic_v3(
    root: str | Path,
    *,
    n_episodes: int = 12,
    n_frames: int = 12,
    frame_hw: tuple[int, int] = (16, 16),
    state_dim: int = 4,
    action_dim: int = 2,
    fps: int = 10,
    seed: int = 0,
    dup_groups: list[list[int]] | None = None,
    mislabel_eps: list[int] | None = None,
    empty_task_eps: list[int] | None = None,
) -> SyntheticGroundTruth:
    """Write a synthetic dataset and return its ground truth."""
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    h, w = frame_hw

    base_tasks = ["pick up the cube", "open the drawer", "stack the blocks", "pour the cup"]
    dup_groups = dup_groups or []
    mislabel_eps = mislabel_eps or []
    empty_task_eps = empty_task_eps or []

    # canonical episode source for duplicate groups: map each member to its leader
    dup_leader: dict[int, int] = {}
    dup_pairs: list[tuple[int, int]] = []
    for group in dup_groups:
        leader = group[0]
        for member in group[1:]:
            dup_leader[member] = leader
            dup_pairs.append((leader, member))

    # pre-generate frames/state/action per leader so duplicates are identical
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def gen(ep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.random.RandomState(seed * 1000 + ep)
        frames = _episode_frames(r, n_frames, frame_hw)
        state = np.cumsum(r.randn(n_frames, state_dim) * 0.1, axis=0).astype(np.float32)
        action = (r.randn(n_frames, action_dim) * 0.1).astype(np.float32)
        return frames, state, action

    gt_tasks: dict[int, str] = {}
    episodes_meta: list[dict[str, object]] = []
    for ep in range(n_episodes):
        leader = dup_leader.get(ep, ep)
        if leader not in cache:
            cache[leader] = gen(leader)
        frames, state, action = cache[leader]

        # task assignment + mislabel/empty injection
        true_task = base_tasks[ep % len(base_tasks)]
        if ep in empty_task_eps:
            task = ""
        elif ep in mislabel_eps:
            # deliberately wrong: a task from a different family
            task = base_tasks[(ep + 2) % len(base_tasks)]
        else:
            task = true_task
        gt_tasks[ep] = task

        rows = {
            "observation.state": [state[t].tolist() for t in range(n_frames)],
            "action": [action[t].tolist() for t in range(n_frames)],
            "frame": [frames[t].ravel().tolist() for t in range(n_frames)],
            "timestamp": [float(t) / fps for t in range(n_frames)],
            "episode_index": [ep] * n_frames,
            "frame_index": list(range(n_frames)),
        }
        table = pa.table(rows)
        pq.write_table(table, root / "data" / f"episode_{ep:06d}.parquet")  # type: ignore[no-untyped-call]
        episodes_meta.append({"episode_index": ep, "length": n_frames, "tasks": [task]})

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "synthetic",
        "fps": fps,
        "total_episodes": n_episodes,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
            "frame": {"dtype": "float32", "shape": [h, w, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (root / "meta" / "episodes.jsonl").open("w") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")

    return SyntheticGroundTruth(
        n_episodes=n_episodes,
        frame_hw=frame_hw,
        dup_pairs=dup_pairs,
        mislabel_eps=sorted(mislabel_eps),
        empty_task_eps=sorted(empty_task_eps),
        tasks=gt_tasks,
    )
