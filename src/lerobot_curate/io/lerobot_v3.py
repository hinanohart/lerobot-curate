"""Torch-free reader for LeRobot v3 datasets (local path or HF Hub repo id).

Only ``huggingface_hub`` + ``pyarrow`` are needed to read metadata and the
synthetic on-disk layout used by tests and the materialize smoke. Decoding real
Hub video frames needs the optional ``[video]`` (PyAV) extra and network access,
which is why frame decoding for video-backed datasets is marked ``live`` and is
never run in CI. Episode *metadata* (refs, tasks, lengths) is always available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..ir import EpisodeRef


class LeRobotV3Reader:
    def __init__(
        self, root: str, revision: str | None = None, cache_dir: str | None = None
    ) -> None:
        self.root = str(root)
        self.revision = revision
        self.cache_dir = cache_dir
        self._is_local = Path(self.root).exists()
        self._info: dict[str, Any] | None = None
        self._episodes: list[dict[str, Any]] | None = None

    # -- path resolution -----------------------------------------------------
    def _meta_path(self, rel: str) -> str:
        if self._is_local:
            return str(Path(self.root) / rel)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            self.root, rel, repo_type="dataset", revision=self.revision, cache_dir=self.cache_dir
        )

    def is_local(self) -> bool:
        return self._is_local

    def info(self) -> dict[str, Any]:
        if self._info is None:
            self._info = json.loads(Path(self._meta_path("meta/info.json")).read_text())
        assert self._info is not None
        return self._info

    @property
    def fps(self) -> float:
        return float(self.info().get("fps", 10))

    def _load_episodes(self) -> list[dict[str, Any]]:
        if self._episodes is not None:
            return self._episodes
        # Prefer episodes.jsonl (our synthetic layout / older v3); fall back to parquet.
        try:
            path = self._meta_path("meta/episodes.jsonl")
            rows = [
                json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
            ]
        except Exception:  # noqa: BLE001
            path = self._meta_path("meta/episodes.parquet")
            tbl = pq.read_table(path).to_pylist()  # type: ignore[no-untyped-call]
            rows = list(tbl)
        self._episodes = sorted(rows, key=lambda r: int(r["episode_index"]))
        return self._episodes

    @staticmethod
    def _first_task(row: dict[str, Any]) -> str | None:
        tasks = row.get("tasks")
        if isinstance(tasks, list) and tasks:
            return str(tasks[0])
        if isinstance(tasks, str):
            return tasks
        return None

    def episode_refs(self) -> list[EpisodeRef]:
        out = []
        for row in self._load_episodes():
            ep = int(row["episode_index"])
            out.append(
                EpisodeRef(
                    repo_id=self.root,
                    episode_index=ep,
                    num_frames=int(row.get("length", 0)),
                    fps=self.fps,
                    task=self._first_task(row),
                )
            )
        return out

    def episode_task(self, ep: int) -> str | None:
        for row in self._load_episodes():
            if int(row["episode_index"]) == ep:
                return self._first_task(row)
        return None

    # -- data ----------------------------------------------------------------
    def _data_path(self, ep: int) -> str:
        return self._meta_path(f"data/episode_{ep:06d}.parquet")

    def _read_episode_table(self, ep: int) -> dict[str, list[Any]]:
        cols: dict[str, list[Any]] = pq.read_table(self._data_path(ep)).to_pydict()  # type: ignore[no-untyped-call]
        return cols

    def episode_frames(self, ep: int, max_frames: int = 8) -> np.ndarray:
        """Representative frames for an episode, shape ``(k, H, W, 3)``.

        For the synthetic/parquet layout, frames are stored as flattened float
        rows. Video-backed Hub datasets require the ``[video]`` extra (live).
        """
        info = self.info()
        feat = info.get("features", {}).get("frame")
        cols = self._read_episode_table(ep)
        if "frame" not in cols:
            raise NotImplementedError(
                "this dataset stores frames as video; decode requires the [video] "
                "extra (PyAV) and is not available in this environment"
            )
        shape = tuple(feat["shape"]) if feat else None
        frames = cols["frame"]
        n = len(frames)
        idx = np.unique(np.linspace(0, n - 1, min(max_frames, n)).astype(int))
        out = []
        for i in idx:
            arr = np.asarray(frames[i], dtype=np.float32)
            if shape is not None:
                arr = arr.reshape(shape)
            out.append(arr)
        return np.stack(out)

    def episode_path(self, ep: int) -> np.ndarray:
        """State+action time series for the signature path, shape ``(T, s+a)``."""
        cols = self._read_episode_table(ep)
        parts = []
        if "observation.state" in cols:
            parts.append(np.asarray(cols["observation.state"], dtype=float))
        if "action" in cols:
            parts.append(np.asarray(cols["action"], dtype=float))
        if not parts:
            raise ValueError(f"episode {ep} has no state/action columns for a signature path")
        return np.concatenate(parts, axis=1)

    def episode_actions(self, ep: int) -> np.ndarray | None:
        """Action time series ``(T, action_dim)`` if present, else ``None``."""
        cols = self._read_episode_table(ep)
        if "action" not in cols:
            return None
        return np.asarray(cols["action"], dtype=float)


def open_dataset(
    root: str, revision: str | None = None, cache_dir: str | None = None
) -> LeRobotV3Reader:
    """Open a LeRobot v3 dataset from a local path or a Hugging Face repo id."""
    return LeRobotV3Reader(root, revision=revision, cache_dir=cache_dir)
